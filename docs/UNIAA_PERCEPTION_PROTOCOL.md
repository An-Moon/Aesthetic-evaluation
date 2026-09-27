# UNIAA Perception protocol

The prepared benchmark contains 5,354 multiple-choice questions with the
official prompt form `Choose between one of the options as follows`. Generation
is deterministic greedy decoding with 1,024 new-token capacity, one beam, and
cache enabled.

`strict-v2` recognizes only unambiguous option letters or a uniquely normalized
option text. Substring containment is forbidden: for example, `Balanced` cannot
match `Unbalanced`, and `Blue` cannot match `Light blue`. Conflicting or
ambiguous answers are invalid, never guessed.

Primary reporting includes overall accuracy over all 5,354 questions, valid
rate, invalid count, breakdowns by aesthetic dimension, question type, and
source dataset, plus paired exact tests on shared questions with Holm correction.
Accuracy uses the full benchmark denominator, so invalid responses count as
incorrect.

The protocol matches the official UNIAA prompt/decode boundary. Cross-model
adapters still use each checkpoint's native image preprocessing and chat
template; those implementation differences are listed in the model configs.
