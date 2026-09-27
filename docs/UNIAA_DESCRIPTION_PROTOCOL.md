# UNIAA Description protocol

The official Description subset contains 501 image/reference pairs. Original
index 0 is deterministically fixed as a text-only demonstration; the remaining
500 images are the test set. The example image itself is never supplied and is
not present in the test set.

Selection uses no random seed. The public manifest records the original index,
source image ID, image/reference hashes, derived test-index hash, and prepared
file hashes. The reference text is not redistributed.

All models use the same unified cross-model protocol: greedy decoding,
`do_sample=false`, `num_beams=1`, `max_new_tokens=256`, cache enabled, one image
per test row, and exact 500/500 coverage. This is not the UNIAA paper's native
stochastic Description inference protocol and must not be labeled an official
paper-number reproduction.
