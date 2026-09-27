# Experimental Score Evaluation Toolbox

This subpackage provides an auditable interface for aesthetic score prediction
and regression metrics. It is included in `v0.1.0` as **experimental**: AVA and
TAD headline results are not part of this release, and generic prompt-numeric
scores must not be presented as official model scoring methods.

## Interfaces

```bash
python run.py infer-score --base-config configs/base_score.yaml \
  --model-config configs/models/artimuse.yaml
python run.py eval-score --pred-file outputs/run/predictions.jsonl \
  --output-file outputs/run/metrics.json
```

The configuration requires `AESTHETIC_DATA_ROOT`, `AESTHETIC_MODEL_ROOT` and
`AESTHETIC_OUTPUT_ROOT`. Missing variables, missing images, invalid scores,
parse failures and incomplete batches are fatal; no midpoint or zero fallback
is applied.

## Status boundaries

| Method | Status |
| --- | --- |
| ArtiMuse `score()` | official model API, pending benchmark reproduction |
| UniPercept reward `score()` | official model API, pending benchmark reproduction |
| ArtQuant/Q-Align/Q-SiT WA5 logits | upstream-aligned mapping, pending full validation |
| AesExpert WA5 logits | fallback, not an official regression method |
| Qwen/InternVL/LLaVA numeric prompt | generic experimental baseline |

Model checkpoints, upstream repositories, AVA/TAD annotations and predictions
are not distributed. See `docs/EXTERNAL_DEPENDENCIES.md` and the root `NOTICE`.
