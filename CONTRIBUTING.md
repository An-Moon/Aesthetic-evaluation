# Contributing

Contributions must preserve protocol provenance and fail-loud behavior.

1. Create a focused branch and add tests for behavior changes.
2. Do not commit checkpoints, datasets, complete predictions, credentials or
   machine-specific absolute paths.
3. New adapters must document the official repository, checkpoint revision,
   preprocessing, conversation template, precision and generation settings.
4. New metrics must define their scale, aggregation, model revision and failure
   behavior. Metric exceptions must not be converted to zero scores.
5. Run `python -m unittest discover -s tests -v` and
   `python scripts/check_release_integrity.py` before opening a pull request.

By contributing, you agree that your contribution is licensed under the
Apache License 2.0 used by this project.
