# Data boundary and local layout

No benchmark image, annotation, reference answer, or prepared full JSONL is
redistributed. Download data from the owners and comply with their terms:

- ArtiMuse-10K: <https://huggingface.co/datasets/Thunderbolt215215/ArtiMuse-10K>
- UNIAA: <https://github.com/KlingAIResearch/Uniaa>

ArtiMuse-10K explicitly prohibits redistribution. This repository publishes
only counts, SHA-256 hashes, source IDs for deterministic few-shot selection,
and preparation code.

Expected local layout below `${AESTHETIC_DATA_ROOT}`:

```text
Artimuse/
  images/
  text/train.jsonl
  text/test.jsonl
UNIAA/A_UNIAA_Bench/
  description/
  perception/
prepared/
  uniaa_description_1shot/test_500.jsonl
  uniaa_perception/test_5354.jsonl
```

The authoritative counts and hashes are in
`results/v0.1.0/data_manifest.json`. Hash mismatch is a protocol mismatch, not a
warning to ignore.
