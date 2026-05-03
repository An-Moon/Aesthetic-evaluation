# TODO: Incremental Score Toolbox Build

## Phase 1: Foundation (in progress)

- [x] Create dedicated score-only repository scaffold.
- [x] Implement infer-score / eval-score / report CLI.
- [x] Add validate-score one-command smoke/validation path.
- [x] Define unified score input and output protocol.
- [x] Extend output protocol with raw_response / parse_status / score_method / official_alignment.
- [x] Integrate ArtiMuse official score path.
- [x] Stop counting prompt parse fallback midpoint values as valid predictions.
- [ ] Add fixed AVA-8000 sample manifest generator.

## Phase 2: Official model integrations

- [x] Integrate OneAlign / Q-Align official logits mapping scorer.
- [x] Integrate ArtQuant official score mapping path.
- [x] Add Q-SiT WA5 scorer path.
- [ ] Integrate UNIAA assessment path.
- [x] Clarify and integrate UniPercept score path.
- [x] Clarify and integrate AesExpert fallback logits-WA5 path.

## Phase 3: General MLLM adapters

- [x] Add generic prompt-numeric adapter.
- [ ] Add Qwen3VL dedicated adapter (if behavior differs).
- [ ] Add InternVL3.5 dedicated adapter (if behavior differs).
- [ ] Add LLaVA dedicated adapter (if behavior differs).

## Phase 4: Validation and reproducibility

- [ ] Run AVA-8000 validation for each integrated model.
- [x] Store per-run sample id list for reproducibility.
- [ ] Add environment snapshot and restore scripts.
- [x] Document mirror endpoint for Hugging Face access.

## Notes

- vLLM 30B path is intentionally excluded in the current phase.
- Main `llava` env supports current ArtiMuse/ArtQuant/AesExpert/UniPercept/OneAlign paths; Q-SiT and LLaVA-OneVision currently require `vllm_qwen` because `llava` has Transformers 4.37.2 and lacks newer LLaVA-OneVision Transformers internals.
- Monitoring details and latest test status are tracked in TODO_MONITOR.md.
