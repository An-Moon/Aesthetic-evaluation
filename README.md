# Aesthetic Eval Framework

A unified, reproducible evaluation framework for large-scale aesthetic VLM testing.

## Goals

- Unified preprocessing: fixed image resize to 448x448.
- Unified prompt and generation parameter management via YAML.
- Unified output protocol: inference first, offline metrics later.
- Adapter-based model integration for InternVL, Qwen3-VL, and LLaVA-OneVision.
- Reproducible runs with config snapshots and runtime metadata.

## Project Layout

- `run.py`: CLI entrypoint (`infer` and `eval`).
- `configs/base.yaml`: shared experiment configuration.
- `configs/models/*.yaml`: model-specific adapter and loading config.
- `src/aesthetic_eval/data.py`: dataset parsing and image loading.
- `src/aesthetic_eval/inference.py`: batched inference with CPU/GPU pipeline overlap.
- `src/aesthetic_eval/metrics.py`: offline metric computation.
- `src/aesthetic_eval/adapters/*`: model adapters.
- `scripts/*.sh`: environment-aware launch scripts.

## Install

Use per-model environments as requested:

- `qwen_vl`: Qwen + InternVL
- `llava`: LLaVA

Install common evaluation deps in each environment as needed:

```bash
pip install -r requirements-common.txt
```

## Hugging Face Mirror

All scripts in `scripts/` now default to:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

If you already set `HF_ENDPOINT` in your shell, scripts will keep your value.

## Environment Snapshot And Reuse

To support disk cleanup and later re-creation, snapshots are stored under:

`/home/Hu_xuanwei/aesthetic_eval_framework/env_snapshots/`

Latest snapshot directory in this run:

`/home/Hu_xuanwei/aesthetic_eval_framework/env_snapshots/20260403_163418`

Each environment has 3 files:

- `qwen_vl.environment.yml` / `llava.environment.yml`: conda environment yaml.
- `qwen_vl.explicit.txt` / `llava.explicit.txt`: exact conda package list.
- `qwen_vl.pip-freeze.txt` / `llava.pip-freeze.txt`: pip package lock.

### Recreate Environment (recommended)

```bash
conda env create -n qwen_vl -f /home/Hu_xuanwei/aesthetic_eval_framework/env_snapshots/20260403_163418/qwen_vl.environment.yml
conda env create -n llava -f /home/Hu_xuanwei/aesthetic_eval_framework/env_snapshots/20260403_163418/llava.environment.yml
```

### Recreate With Explicit Lock (stricter)

```bash
conda create -n qwen_vl --file /home/Hu_xuanwei/aesthetic_eval_framework/env_snapshots/20260403_163418/qwen_vl.explicit.txt
conda create -n llava --file /home/Hu_xuanwei/aesthetic_eval_framework/env_snapshots/20260403_163418/llava.explicit.txt
```

### Reapply pip lock (optional)

```bash
conda run -n qwen_vl pip install -r /home/Hu_xuanwei/aesthetic_eval_framework/env_snapshots/20260403_163418/qwen_vl.pip-freeze.txt
conda run -n llava pip install -r /home/Hu_xuanwei/aesthetic_eval_framework/env_snapshots/20260403_163418/llava.pip-freeze.txt
```

### Safe Cleanup Pattern (repeatable)

1. Export snapshots to a timestamped directory.
2. Verify snapshots exist (`*.environment.yml`, `*.explicit.txt`, `*.pip-freeze.txt`).
3. Remove only target envs:

```bash
conda remove -n qwen_vl --all -y
conda remove -n llava --all -y
```

4. Verify remaining envs with `conda env list`.

### One-Command Pattern Script

Use this script for repeated operations:

```bash
cd /home/Hu_xuanwei/aesthetic_eval_framework
bash scripts/env_snapshot_and_cleanup.sh qwen_vl llava
```

The script will:

1. Create a timestamped snapshot directory in `env_snapshots/`.
2. Export `environment.yml`, `explicit.txt`, and `pip-freeze.txt` for each env.
3. Remove only the envs you pass as arguments.
4. Print remaining environments for verification.

## Inference

```bash
cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=0 /home/Hu_xuanwei/.conda/envs/qwen_vl/bin/python run.py infer \
  --base-config configs/base.yaml \
  --model-config configs/models/internvl.yaml

cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=0 /home/Hu_xuanwei/.conda/envs/llava/bin/python run.py infer \
  --base-config configs/base.yaml \
  --model-config configs/models/llava.yaml

cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=0 /home/Hu_xuanwei/.conda/envs/artimuse/bin/python run.py infer \
  --base-config configs/base.yaml \
  --model-config configs/models/artimuse.yaml

cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=0 /home/Hu_xuanwei/.conda/envs/artimuse/bin/python run.py infer \
  --base-config configs/base.yaml \
  --model-config configs/models/uniaa.yaml

cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=0 /home/Hu_xuanwei/.conda/envs/artquant/bin/python run.py infer \
  --base-config configs/base.yaml \
  --model-config configs/models/artquant.yaml

cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=0 /home/Hu_xuanwei/.conda/envs/aesexpert/bin/python run.py infer \
  --base-config configs/base.yaml \
  --model-config configs/models/aesexpert.yaml

cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=0 /home/Hu_xuanwei/.conda/envs/onealign/bin/python run.py infer \
  --base-config configs/base.yaml \
  --model-config configs/models/onealign.yaml

cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=0 /home/Hu_xuanwei/.conda/envs/artquant/bin/python run.py infer \
  --base-config configs/base.yaml \
  --model-config configs/models/qsit.yaml

# Note: qsit.yaml now defaults to infer_mode: generate for description tasks.

cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=0 /home/Hu_xuanwei/.conda/envs/artimuse/bin/python run.py infer \
  --base-config configs/base.yaml \
  --model-config configs/models/unipercept.yaml


cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=3 /home/Hu_xuanwei/.conda/envs/qwen_vl/bin/python run.py infer \
  --base-config configs/base.yaml \
  --model-config configs/models/qwen3_vl_lora_ft.yaml
```

## Offline Metrics

```bash
cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=3 /home/Hu_xuanwei/.conda/envs/llava/bin/python run.py eval \
  --pred-file outputs/qwen3_vl_lora_ft_description_20260406_094730/predictions.jsonl \
  --output-file outputs/description/metrics_qwenLora_summary.json

# Safe mode for CLIP model (prevent long hang):
cd /home/Hu_xuanwei/aesthetic_eval_framework && CUDA_VISIBLE_DEVICES=0 /home/Hu_xuanwei/.conda/envs/qwen_vl/bin/python run.py eval \
  --pred-file outputs/internvl_description_20260403_123752/predictions.jsonl \
  --output-file outputs/qwen_description/metrics_summary.json \
  --enabled bleu rouge meteor bertscore sbert_cos clipscore \
  --clip-model openai/clip-vit-base-patch32 \
  --clip-timeout 120 \
  --clip-local-only
```

You can also wrap eval command with shell `timeout`, e.g. `timeout 30m ...`, to avoid unreasonable long stalls.

## Aesthetic Models

- ArtiMuse config: `configs/models/artimuse.yaml`
- UNIAA config: `configs/models/uniaa.yaml`
- ArtQuant config: `configs/models/artquant.yaml`
- AesExpert config: `configs/models/aesexpert.yaml`
- OneAlign config: `configs/models/onealign.yaml`
- Q-SiT config: `configs/models/qsit.yaml`
- UniPercept config: `configs/models/unipercept.yaml`
- ArtiMuse script: `scripts/run_infer_artimuse.sh`
- UNIAA script: `scripts/run_infer_uniaa.sh`
- ArtQuant script: `scripts/run_infer_artquant.sh`
- AesExpert script: `scripts/run_infer_aesexpert.sh`
- OneAlign script: `scripts/run_infer_onealign.sh`
- Q-SiT script: `scripts/run_infer_qsit.sh`
- UniPercept script: `scripts/run_infer_unipercept.sh`
- Chinese integration TODO: `美学模型接入说明与TODO.md`

## Output Protocol

Inference writes:

- `predictions.jsonl`: one record per sample.
- `run_meta.json`: config snapshot, timing, environment info.

Each prediction row includes:

- `sample_id`
- `image`
- `image_resolved`
- `prompt`
- `prediction`
- `reference`
- `model`
- `task`
- `timestamps`

This enables adding new metrics later without rerunning inference.
# Aesthetic-evaluation
