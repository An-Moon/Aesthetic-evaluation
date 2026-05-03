# Release Audit

Date: 2026-05-03

## Checked Items

- External official repository dependencies are documented in `docs/EXTERNAL_DEPENDENCIES.md`.
- Dataset, output, checkpoint, and official repo paths are configurable through environment variables.
- README contains quick start, CLI examples, data format, output protocol, model status, and environment policy.
- Requirements are split into common CLI dependencies and model-family runtime files.
- `.gitignore` excludes outputs, caches, virtual environments, model weights, and checkpoints.
- CLI help and YAML environment expansion were checked locally.

## Remaining Before Public Release

- Add a project license.
- Add citation/BibTeX once paper metadata is final.
- Replace `data/ava_smoke_64.jsonl` with a real local smoke subset before running smoke tests.
- Run a clean-clone smoke test with only `.env` paths configured.
- Keep official upstream repositories and model checkpoints out of git.

## Note On The VQA Repository

The separate long-text VQA evaluation repository still contains many local
absolute paths in README, scripts, configs, and adapters. It needs a similar
portability pass before public release.
