# Model and environment matrix

The machine-readable matrix is `results/v0.1.0/model_matrix.csv`. Status means:

- `validated`: complete output passed coverage and behavior checks.
- `experimental`: adapter/tooling is public but lacks a headline result.
- `invalid_for_protocol`: outputs failed a predefined validity gate and are not
  used in primary tables.
- `in_domain_only`: result is reported separately because training/evaluation
  domain overlap would bias the main comparison.

Adapters always use local checkpoints. Example YAML files pin recorded Hub
revisions where that information survived acquisition. `unrecorded_local_snapshot`
is deliberately explicit; it must not be interpreted as a floating `main`
revision. Environment lock files are separated because LLaVA-family and newer
Qwen/InternVL checkpoints require incompatible Transformers stacks.

Q-SiT copied the 16-shot demonstration response across distinct smoke-test
images. AesExpert produced invalid Description behavior under the current
adapter/protocol. Their adapters remain available for diagnosis, but neither
model enters primary Description analysis.
