# ArtiMuse Description protocol

The test set contains 7,984 dimension-specific image/question/reference rows,
998 rows for each of eight aesthetic dimensions. Inference validates all image
paths and unique sample IDs before loading a model; final output must be exactly
7,984 unique rows.

## 0-shot

The model receives the current image and its dimension-specific question only.
Generation is deterministic greedy decoding (`do_sample=false`, `num_beams=1`,
`max_new_tokens=256`, cache enabled).

## 16-shot text-only

Sixteen reference examples are prepended as text: two examples for each of the
eight dimensions. No demonstration image is supplied. Test source IDs are
removed from the training candidate pool before selection. Within each
dimension, one item is closest to median answer length; the second minimizes
lexical overlap with the first and then length distance. The public manifest
contains source IDs, order, word counts, and answer hashes, but not answer text.

The 16-shot protocol is context learning, not a free-reference evaluation. The
same deterministic decoding controls are used as in 0-shot.

## Reporting boundary

The ArtiMuse checkpoint is trained on ArtiMuse-10K and is therefore marked
`in_domain_only`; it is not used in the primary General-versus-IAA group
comparison. Q-SiT and AesExpert fail the predefined output-validity boundary and
are excluded. ArtQuant remains, with its template-like behavior disclosed.
