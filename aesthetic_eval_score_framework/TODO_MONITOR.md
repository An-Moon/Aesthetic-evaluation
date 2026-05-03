# TODO Monitor: Score Toolbox Stability Gate

## Latest smoke run (ArtiMuse official score path)

- Date: 2026-04-29
- Command environment: legacy LLaVA-style environment
- GPU target: physical GPU3 via CUDA_VISIBLE_DEVICES=3
- Effective runtime GPU: cuda:0 (masked single visible device, expected)
- Base config: configs/base_score_smoke_gpu3.yaml
- Model config: configs/models/artimuse_gpu3_smoke.yaml
- Output dir: outputs/artimuse_score_20260429_160211

## Smoke sample data

- Input jsonl: data/ava_smoke_64.jsonl
- Loaded samples: 33
- Missing images: 0
- Sample manifest: outputs/ava_smoke_manifest.txt

## Numerical correctness checks (must-pass)

- Inference errors: 0/33
- None values in raw_score/score_0_10: 0
- NaN values in raw_score/score_0_10: 0
- raw_score in [0, 100]: pass (33/33)
- score_0_10 in [0, 10]: pass (33/33)
- Linear mapping consistency (score_0_10 == raw_score / 10): pass (33/33)

## Regression metrics (smoke only, not benchmark conclusion)

- N: 33
- PLCC: 0.18900592178013623
- SRCC: 0.30477307685813493
- KROCC: 0.2074238506146593
- MAE: 0.8311760549834282
- RMSE: 1.150117450609419

## Environment compatibility findings

1. qwen_vl env is incompatible for ArtiMuse loading in current state:
   - transformers 5.5.4 triggers model loading incompatibility.
2. llava env is compatible after adapter-side inference shim:
   - Added ArtiMuse config compatibility flag for newer transformers paths.
   - Added peft fallback shim when peft/accelerate import chain is broken.
3. FlashAttention is not required for this smoke path:
   - Config uses use_flash_attn: false for stability.

## Gate before integrating next model

- [x] Current model can run end-to-end on GPU3.
- [x] Numerical correctness checks all pass.
- [x] Sample manifest is generated for reproducibility.
- [ ] Add environment snapshot script for the chosen stable env.
- [ ] Run AVA-8000 validation for ArtiMuse in the same env.
- [ ] Record official-vs-toolbox delta report (PLCC/SRCC diff <= 0.03 target).

## Next integration order (after gate pass)

1. Q-Align official logits scorer
2. ArtQuant official mapping path
3. Q-SiT WA5 path
4. UNIAA assessment path

## Latest smoke run (AesExpert fallback logits-WA5)

- Date: 2026-05-01
- Command environment: legacy LLaVA-style environment
- Extra environment: `HF_ENDPOINT=https://hf-mirror.com`
- GPU target: physical GPU3 via CUDA_VISIBLE_DEVICES=3
- Base config: configs/base_score_smoke_gpu3.yaml
- Model config: configs/models/aesexpert_gpu3_smoke.yaml
- Output dir: outputs/AesExpert_score_20260501_091947

## AesExpert smoke checks

- Loaded samples: 33 (the current smoke jsonl contains 33 resolvable samples)
- Inference errors: 0/33
- Parse failures: 0/33
- Unique score_0_10 values: 33/33
- score_0_10 min/max: 2.720315456390381 / 8.12479019165039
- PLCC/SRCC: 0.19420970936931453 / 0.2168352609352721

## AesExpert findings

- The old prompt-numeric run is no longer considered valid for benchmarking because most rows collapsed to the midpoint fallback.
- The new default is `fallback_logits_wa5`, marked as non-official but auditable.
- Hugging Face mirror access is required unless `openai/clip-vit-large-patch14-336` is already fully cached.

## Latest smoke run (UniPercept official reward score)

- Date: 2026-05-01
- Command environment: legacy LLaVA-style environment
- GPU target: physical GPU3 via CUDA_VISIBLE_DEVICES=3
- Base config: configs/base_score_smoke_gpu3.yaml
- Model config: configs/models/unipercept_gpu3_smoke.yaml
- Output dir: outputs/UniPercept_score_20260501_095645

## UniPercept smoke checks

- Loaded samples: 33
- Inference errors: 0/33
- Parse failures: 0/33
- Unique score_0_10 values: 33/33
- score_0_10 min/max: 2.3751846313476563 / 8.208162689208985
- PLCC/SRCC: 0.2107982489896307 / 0.25545433979113014

## UniPercept findings

- The local HF remote-code file did not expose `score()` in our test checkpoint, so the score adapter intentionally imports the official repo implementation from `UNIPERCEPT_REPO/src`.
- The official score path returns a 0-100 reward-style IAA aesthetics score; the toolbox maps it linearly to 0-10.
- FlashAttention is optional for this smoke path; the stable config uses `use_flash_attn: false`.

## Latest smoke run (OneAlign / Q-Align official aesthetic scorer)

- Date: 2026-05-01
- Command environment: legacy LLaVA-style environment
- GPU target: physical GPU3 via CUDA_VISIBLE_DEVICES=3
- Base config: configs/base_score_smoke_gpu3.yaml
- Model config: configs/models/onealign_gpu3_smoke.yaml
- Output dir: outputs/OneAlign_score_20260501_105157

## OneAlign smoke checks

- Loaded samples: 33
- Inference errors: 0/33
- Parse failures: 0/33
- Unique score_0_10 values: 31/33
- score_0_10 min/max: 2.9833984375 / 7.6806640625
- PLCC/SRCC: 0.7580754231392663 / 0.7892316944174319

## OneAlign findings

- The adapter uses Q-Align official `QAlignAestheticScorer`, returning a raw 0-1 WA5 score mapped to 0-10.
- Q-Align official code hardcodes `q-future/one-align` inside `modeling_mplug_owl2.py`; the adapter redirects that path to `ONEALIGN_MODEL`.
- The adapter also patches the missing `_prepare_4d_causal_attention_mask_for_sdpa` symbol for the current transformers/Q-Align compatibility pair.

## Q-SiT environment gate

- Adapter/config added: `configs/models/qsit.yaml` and `configs/models/qsit_gpu3_smoke.yaml`.
- Main `llava` env Transformers: 4.37.2.
- Q-SiT required class: `LlavaOnevisionForConditionalGeneration`.
- `llava` env status: missing required class, cannot load Q-SiT without upgrading Transformers.
- `vllm_qwen` env Transformers: 4.57.6, required class is available and local weights load until CUDA initialization.
- Decision: keep Q-SiT on `vllm_qwen` unless a cloned unified env is created and all already-integrated models are revalidated there.

## LLaVA-OneVision environment gate

- Adapter/config added: `configs/models/llava_onevision_8b.yaml` and `configs/models/llava_onevision_8b_gpu3_smoke.yaml`.
- Original local model path may contain dots/hyphens, e.g. `LLaVA-OneVision-1.5-8B-Instruct`.
- Safe alias path recommended for older dynamic-module importers, e.g. `LLaVA_OneVision_1_5_8B_Instruct`.
- `llava` env status: too old for this model. After the path issue, remote code still requires newer Transformers internals such as `modeling_rope_utils` and `SlidingWindowCache`.
- `vllm_qwen` env status: smoke passed on GPU0.
- Smoke output dir: `outputs/LLaVA-OneVision-1.5-8B-Instruct_score_20260501_112314`.
- Smoke checks: 4 samples, inference errors 0/4, parse failures 0/4, PLCC/SRCC 0.4431 / 0.6325.
- Decision: run LLaVA-OneVision through `vllm_qwen`; the adapter now raises a clear error when launched from older Transformers environments.
