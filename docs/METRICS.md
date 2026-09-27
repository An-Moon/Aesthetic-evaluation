# Description metrics

All public values use the `[0,1]` scale and are computed against a single
reference per sample.

- **BLEU**: corpus BLEU-4 from SacreBLEU with effective order; n-gram precisions
  and brevity penalty are also retained privately.
- **ROUGE-L**: macro mean of sentence-level longest-common-subsequence F1 with
  stemming.
- **METEOR**: macro mean from NLTK tokenized word sequences.
- **BERT-F1**: BERTScore F1 using `microsoft/deberta-base-mnli` at the pinned
  revision and `num_layers=9`.
- **SBERT-Cos**: cosine similarity between normalized
  `sentence-transformers/all-mpnet-base-v2` sentence embeddings.
- **SPICE**: scene-graph tuple F-score from `pycocoevalcap` using Java 8. It is
  chunked and checkpointed, but any failed chunk terminates the evaluation.
- **CLIP-Cos**: cosine similarity between normalized text and image embeddings
  from `openai/clip-vit-base-patch32`. Text is truncated at the model's 77-token
  limit. This is not the scaled metric named CLIPScore.

The public metric YAML pins model revisions and requires local directories. No
metric downloads a substitute model and no exception is converted to `0.0`.

## Length adjustment

The protocol additionally computes `LP-metric = metric_i * min(Lp,Lr) /
max(Lp,Lr)` before macro averaging where per-sample values exist. Raw metrics
are always preserved and are the values in primary release tables. Length-
adjusted results are diagnostic only.

## Random mismatch diagnostic

The Sattolo protocol creates a derangement for each seed, guaranteeing zero
fixed reference assignments. Five seeds are summarized by arithmetic mean and
sample standard deviation. The released historical UNIAA mismatch run used all
501 Description rows, whereas the model comparison reserves one row and tests
500; it is therefore a diagnostic of dataset/reference structure, not a matched
model baseline.
