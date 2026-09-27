# v0.1.0 aggregate artifacts

This directory is safe for public distribution: it contains no full prediction,
prompt, reference text, image path, answer key, or per-sample detail.

| File | Content |
|---|---|
| `description_overall.csv` | Primary ArtiMuse and UNIAA Description results |
| `artimuse_description_by_dimension.csv` | Eight-dimension aggregate results |
| `description_provenance.csv` | Metric-artifact and prediction SHA-256 values |
| `uniaa_qa_overall.csv` | QA accuracy, valid rate, counts, and hashes |
| `uniaa_qa_subgroups.csv` | QA dimension, question-type, and source aggregates |
| `uniaa_qa_paired_significance.csv` | Exact paired tests with Holm correction |
| `uniaa_description_mismatch_5seed.json` | Five-seed derangement diagnostic |
| `data_manifest.json` | Counts, dataset hashes, and deterministic split metadata |
| `model_matrix.csv` | Model status and checkpoint revision availability |
| `release_manifest.json` | Protocol/config/parser/metric provenance |
| `tables/` | Single-column LaTeX tables |
| `figures/` | Editable SVG and publication PDF figures |

All Description columns are raw metrics. No length-penalized metric is used in
the headline tables. `CLIP-Cos` is normalized embedding cosine, not CLIPScore.

The mismatch diagnostic has 501 rows per seed because it predates the fixed-
first 500-row test split. It must not be treated as a matched random baseline for
the 500-row model table.
