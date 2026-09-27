# Results boundary

`results/v0.1.0` contains only aggregate CSV/JSON, LaTeX tables, and plots. It
does not contain full predictions, references, absolute paths, prompts, or
per-sample QA details.

Description files report 7,984/7,984 ArtiMuse rows and 500/500 UNIAA rows.
UNIAA QA files report 5,354/5,354 for every model, including invalid-response
counts and valid rates. `uniaa_qa_paired_significance.csv` uses the same 5,354
sample IDs for every pair, a two-sided exact binomial McNemar test on discordant
pairs, and Holm correction over all released pairwise comparisons.

Do not compare these unified-protocol Description values to the UNIAA paper's
native stochastic generation results as if the protocols were identical. Do
not infer that lexical, embedding, scene-graph, and image-text metrics measure a
single latent notion of aesthetic reasoning; the release keeps them separate.
