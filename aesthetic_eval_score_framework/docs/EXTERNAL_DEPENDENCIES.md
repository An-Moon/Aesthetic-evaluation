# External Dependencies

This repository owns the unified evaluation CLI, data protocol, metrics, report
builder, and model adapters. It does not vendor full upstream model repositories
or model checkpoints.

## Adapter Dependency Matrix

| Adapter | Model config | External official repo needed? | Why |
| --- | --- | --- | --- |
| ArtiMuse | `configs/models/artimuse.yaml` | Yes: `ARTIMUSE_REPO` | Uses official `InternVLChatModel.score()` implementation. |
| ArtQuant | `configs/models/artquant.yaml` | Yes: `ARTQUANT_REPO` | Reuses official mPLUG-Owl2 + LoRA loader and logits scorer. |
| AesExpert | `configs/models/aesexpert.yaml` | Yes: `LLAVA_REPO` | Uses LLaVA v1.5 loader; scoring is toolbox fallback logits-WA5. |
| UniPercept | `configs/models/unipercept.yaml` | Yes: `UNIPERCEPT_REPO` | Local model remote code did not expose `score()`, so official repo source is used. |
| OneAlign / Q-Align | `configs/models/qalign.yaml` | Yes: `QALIGN_REPO` | Reuses official `QAlignAestheticScorer`. |
| Q-SiT | `configs/models/qsit.yaml` | No separate repo in current adapter | Loads local HF-style model with Transformers OneVision class. |
| Qwen3VL | `configs/models/qwen3vl_4b.yaml` | No | Generic prompt-numeric adapter through Transformers remote code. |
| InternVL3.5 | `configs/models/internvl35_8b.yaml` | No | Generic prompt-numeric adapter through Transformers remote code. |
| LLaVA-OneVision 1.5 | `configs/models/llava_onevision_8b.yaml` | No | Generic prompt-numeric adapter through Transformers remote code. |

## Why Not Vendor Official Repositories?

Several official repositories contain large model definitions, local package
assumptions, and their own licenses. Copying them into this repository would
make maintenance and license review harder. The open-source boundary is
therefore:

- This repo: reproducible evaluation workflow and adapters.
- Upstream repos/checkpoints: installed or cloned by the user and pointed to via
  environment variables.

## Required Environment Variables

See `.env.example` for all path variables. The YAML loader supports both
`${VAR}` and `${VAR:-default}` forms.
