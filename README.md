# Aesthetic Evaluation

Reproducible, fail-loud evaluation for multimodal image-aesthetics models. The
`v0.1.0` release covers reference-based Description experiments on ArtiMuse-10K
and UNIAA-Bench, plus the 5,354-question UNIAA perception benchmark. An
independent score-evaluation toolbox is included as **experimental**; this
release does not claim AVA or TAD headline results.

This repository intentionally does **not** distribute datasets, model weights,
full predictions, prompts containing benchmark references, or per-sample metric
details. Public results are aggregate-only and accompanied by hashes.

## Install

```bash
git clone https://github.com/An-Moon/Aesthetic-evaluation.git
cd Aesthetic-evaluation
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Export the five required roots from `.env.example`. YAML values support `${VAR}`
and `~`; an absent variable or unresolved expression raises immediately.
Install `.[metrics]` in the dedicated metric environment and `.[viz]` only when
regenerating figures.

## Fixed protocols

| Protocol | Public config | Coverage | Decoding |
|---|---|---:|---|
| ArtiMuse Description 0-shot | `configs/protocols/artimuse_description_0shot.yaml` | 7,984 | greedy, 256 tokens |
| ArtiMuse Description 16-shot text-only | `configs/protocols/artimuse_description_16shot.yaml` | 7,984 | greedy, 256 tokens |
| UNIAA Description fixed-first 1-shot | `configs/protocols/uniaa_description_1shot_greedy_v2.yaml` | 500 | greedy, 256 tokens |
| UNIAA Perception official-v1 | `configs/protocols/uniaa_perception_official_v1.yaml` | 5,354 | greedy, 1,024 tokens |

All use `do_sample: false`, `num_beams: 1`, deterministic runtime controls, exact
coverage checks, and resume validation. See [`docs/`](docs/) for the prompt,
selection, metric, model, and comparability boundaries.

## Run

Prepare local benchmark files first:

```bash
python scripts/prepare_uniaa_description_1shot_v1.py \
  --source-root "$AESTHETIC_DATA_ROOT/UNIAA/A_UNIAA_Bench/description" \
  --output-dir "$AESTHETIC_DATA_ROOT/prepared/uniaa_description_1shot" \
  --context-output "$UNIAA_1SHOT_CONTEXT_FILE"

python scripts/prepare_uniaa_perception_official_v1.py \
  --source-root "$AESTHETIC_DATA_ROOT/UNIAA/A_UNIAA_Bench/perception" \
  --output-dir "$AESTHETIC_DATA_ROOT/prepared/uniaa_perception"
```

Run inference with an explicit model config:

```bash
python run.py infer \
  --base-config configs/protocols/uniaa_perception_official_v1.yaml \
  --model-config configs/models/examples/qwen3_vl_8b.yaml
```

Resume only into the original run directory:

```bash
python run.py infer \
  --base-config configs/protocols/uniaa_perception_official_v1.yaml \
  --model-config configs/models/examples/qwen3_vl_8b.yaml \
  --resume-dir "$AESTHETIC_OUTPUT_ROOT/uniaa_perception_official_v1/<run>"
```

Description metrics are offline and strict:

```bash
python run.py eval \
  --pred-file "$AESTHETIC_OUTPUT_ROOT/<run>/predictions.jsonl" \
  --output-file "$AESTHETIC_OUTPUT_ROOT/<run>/metrics_v2.json" \
  --metrics-config configs/metrics/artimuse_description_v2.yaml \
  --by-dimension
```

UNIAA QA uses the strict option parser:

```bash
python scripts/eval_uniaa_perception_strict.py \
  --pred-file "$AESTHETIC_OUTPUT_ROOT/<run>/predictions.jsonl" \
  --dataset-file "$AESTHETIC_DATA_ROOT/prepared/uniaa_perception/test_5354.jsonl" \
  --output-file "$AESTHETIC_OUTPUT_ROOT/<run>/accuracy_strict_v2.json"
```

The default QA output is aggregate-only. `--include-details` is available for
private diagnostics and must not be committed.

## Failure policy

There is no fallback to empty text, guessed labels, downloaded substitute
weights, or numeric zero. Missing models, dependencies, environment variables,
images, samples, duplicate IDs, empty predictions, metric timeouts, and parser
ambiguity are either fatal or explicitly counted invalid by the protocol.

The metric named `CLIP-Cos` is the cosine between normalized CLIP text and image
embeddings. It is not the scaled reference-free metric commonly called
CLIPScore. The old input key `clipscore` is accepted for one release with a
deprecation warning; new outputs always use `CLIP-Cos`.

## Results and status

Aggregate artifacts, manifests, tables, and editable figures are under
[`results/v0.1.0/`](results/v0.1.0/). The model matrix distinguishes
`validated`, `experimental`, `invalid_for_protocol`, and `in_domain_only`.
Q-SiT and AesExpert are excluded from primary Description tables due to output
validity failures. ArtiMuse is excluded from the primary ArtiMuse Description
group comparison because it is in-domain. ArtQuant remains included with an
explicit template-output caveat.

Official UNIAA paper numbers, this repository's unified cross-model protocols,
and the custom text-similarity metrics are separate result families and must not
be described as direct reproductions of one another.

## License and citation

Original repository code is licensed under Apache-2.0. External checkpoints,
datasets, and optional upstream repositories retain their own terms; see
[`NOTICE`](NOTICE) and [`docs/DATA.md`](docs/DATA.md). Formal paper citation
metadata is pending, so this release intentionally does not include a
`CITATION.cff`.
