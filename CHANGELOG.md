# Changelog

## Unreleased

- Keeps only final SVG visualizations in the repository; numeric result exports
  and per-sample artifacts remain private.
- Consolidates public usage and protocol guidance in the root README.
- Removes the public test suite while retaining CI smoke, compilation, config,
  and release-boundary checks.

## v0.1.0 - 2026-09-27

- Publishes portable, fail-loud description and UNIAA perception evaluation.
- Adds fixed ArtiMuse 0-shot/16-shot, UNIAA Description 1-shot and UNIAA
  Perception strict-v2 protocols.
- Adds validated and experimental model-status metadata.
- Renames the normalized CLIP image-text cosine metric to `CLIP-Cos`.
- Publishes aggregate-only result artifacts and provenance manifests.
- Adds an explicitly experimental score-evaluation toolbox without AVA/TAD
  headline results.
- Removes runtime outputs, compiled caches, machine-specific paths and dataset
  annotation copies from the public release history.
