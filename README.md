# Aesthetic Evaluation

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://kozmojor.github.io/Aesthetic-evaluation/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-TBD-lightgrey)](#license)

A unified and reproducible benchmark framework for evaluating image aesthetic understanding in vision-language models and aesthetic scoring models.

This repository provides a shared inference protocol, model adapter interface, metric pipeline, and output format for comparing multiple aesthetic evaluation approaches under consistent data preprocessing and reporting settings.

> Project webpage: https://kozmojor.github.io/Aesthetic-evaluation/

## Highlights

- Unified model interface for aesthetic description and score-based evaluation.
- YAML-based experiment configuration for datasets, prompts, generation settings, runtime options, and model adapters.
- Reproducible inference outputs with prediction files, runtime metadata, and config snapshots.
- Offline metric computation, so new metrics can be added without rerunning expensive model inference.
- Adapter-based integration for both general MLLMs and aesthetic-specific models.
- Support for text-generation metrics and score-regression metrics through separate but related evaluation pipelines.

## Supported Models

The current framework contains adapters or launch scripts for:

| Model | Main config/script |
| --- | --- |
| InternVL | `configs/models/internvl.yaml` |
| Qwen / Qwen3-VL | `configs/models/qwen.yaml`, `configs/models/qwen3_vl_lora_ft.yaml` |
| LLaVA-OneVision | `configs/models/llava.yaml` |
| ArtiMuse | `configs/models/artimuse.yaml` |
| UNIAA | `configs/models/uniaa.yaml` |
| ArtQuant | `configs/models/artquant.yaml` |
| AesExpert | `configs/models/aesexpert.yaml` |
| OneAlign / Q-Align | `configs/models/onealign.yaml` |
| Q-SiT | `configs/models/qsit.yaml` |
| UniPercept | `configs/models/unipercept.yaml` |

Model-specific dependencies and checkpoints are not vendored in this repository. Please prepare the official model weights and upstream repositories according to each model's license and usage instructions.

## Repository Layout

```text
.
+-- run.py                         # CLI entrypoint for description inference and evaluation
+-- configs/
|   +-- base.yaml                  # Shared experiment configuration
|   +-- models/                    # Model-specific adapter configs
+-- src/aesthetic_eval/
|   +-- data.py                    # Dataset loading and image resolution
|   +-- inference.py               # Batched inference pipeline
|   +-- metrics.py                 # Offline metric computation
|   +-- adapters/                  # Model adapter implementations
+-- scripts/                       # Environment-aware launch scripts
+-- outputs/                       # Example prediction and metric outputs
+-- aesthetic_eval_score_framework/ # Score-focused evaluation toolbox
```

## Installation

Clone the repository:

```bash
git clone https://github.com/An-Moon/Aesthetic-evaluation.git
cd Aesthetic-evaluation
```

Create an environment and install the common dependencies:

```bash
conda create -n aesthetic-eval python=3.10 -y
conda activate aesthetic-eval
pip install -r requirements-common.txt
```

Some models require separate environments because their official implementations depend on different versions of PyTorch, Transformers, or LLaVA-related packages. The scripts in `scripts/` are intended to be used with the environment that matches each model.

If Hugging Face access is slow in your network environment, set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Data Format

For description-based evaluation, the dataset file should contain image-question-reference examples. The loader accepts JSON/JSONL-style records with fields used by the configured task, such as:

```json
{
  "sample_id": "000001",
  "image": "000001.jpg",
  "question": "Please describe the aesthetic quality of this image.",
  "reference": "The image has balanced composition and harmonious colors."
}
```

Set the dataset path and image root in `configs/base.yaml`:

```yaml
data:
  dataset_json: /path/to/eval.json
  image_root: /path/to/images
  image_alt_root: null
  sample_limit: null
```

For score-based evaluation, see [`aesthetic_eval_score_framework/README.md`](aesthetic_eval_score_framework/README.md).

## Quick Start

Run inference with a shared base config and a model config:

```bash
python run.py infer \
  --base-config configs/base.yaml \
  --model-config configs/models/internvl.yaml
```

Evaluate a completed prediction file:

```bash
python run.py eval \
  --pred-file outputs/<run_name>/predictions.jsonl \
  --output-file outputs/<run_name>/metrics_summary.json
```

You can select a subset of metrics:

```bash
python run.py eval \
  --pred-file outputs/<run_name>/predictions.jsonl \
  --output-file outputs/<run_name>/metrics_summary.json \
  --enabled bleu rouge meteor bertscore sbert_cos clipscore
```

Convenience scripts are available for individual models:

```bash
bash scripts/run_infer_qwen.sh
bash scripts/run_infer_llava.sh
bash scripts/run_infer_artquant.sh
bash scripts/run_eval.sh
```

## Output Protocol

Each inference run writes a timestamped directory under the configured output root. The main files are:

- `predictions.jsonl`: one prediction record per sample.
- `run_meta.json`: model name, task name, config snapshot, timing, and runtime metadata.
- `metrics_summary.json`: offline evaluation result, when `run.py eval` is executed.

Each prediction row follows a unified schema:

```json
{
  "sample_id": "000001",
  "image": "000001.jpg",
  "image_resolved": "/abs/path/to/000001.jpg",
  "prompt": "Please describe the aesthetic quality of this image.",
  "prediction": "...",
  "reference": "...",
  "model": "internvl",
  "task": "description",
  "timestamps": {}
}
```

This design makes inference auditable and lets users recompute metrics later without calling the model again.

## Metrics

The description pipeline currently supports:

- BLEU
- ROUGE
- METEOR
- BERTScore
- Sentence-BERT cosine similarity
- CLIPScore

The score-focused framework supports correlation and regression metrics including PLCC, SRCC, KROCC, MAE, MSE, and RMSE.

## Score Evaluation Toolbox

The [`aesthetic_eval_score_framework`](aesthetic_eval_score_framework/) directory contains a dedicated pipeline for single-image aesthetic score regression. It provides:

- `infer-score`: unified score inference.
- `eval-score`: score metric computation.
- `validate-score`: smoke-test inference plus metrics.
- `report`: leaderboard aggregation.

See [`aesthetic_eval_score_framework/README.md`](aesthetic_eval_score_framework/README.md) for detailed usage.

## Reproducibility Notes

- Keep dataset manifests fixed when comparing models.
- Use the same `configs/base.yaml` for all models in one benchmark table.
- Store model-specific changes only in `configs/models/*.yaml`.
- Keep generated `predictions.jsonl` and `run_meta.json` files together.
- Record checkpoint versions and upstream repository commits for external models.

## Project Page

We also provide a webpage for a more visual overview of the project, including motivation, framework design, and result presentation:

https://kozmojor.github.io/Aesthetic-evaluation/

## Contributing

Contributions are welcome. Useful contributions include:

- New model adapters under `src/aesthetic_eval/adapters/`.
- Additional metric implementations.
- Cleaner dataset converters or validation scripts.
- Documentation improvements and reproducible experiment reports.

When adding a new model, please include:

1. A model config in `configs/models/`.
2. An adapter implementation or a script that follows the unified output protocol.
3. A short note about required checkpoints, upstream repositories, and environment constraints.

## Citation

If this repository is useful for your research or coursework, please cite or acknowledge the project page and repository. A formal citation entry can be added here when the report or paper information is finalized.

## License

The project license has not been specified yet. Before public release, please add a `LICENSE` file and update the badge at the top of this README.
