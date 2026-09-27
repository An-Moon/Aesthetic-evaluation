#!/usr/bin/env python3
"""Fail-loud public-release boundary checks."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import yaml


FORBIDDEN_DIRS = {
    "outputs", "new_outputs", "model", "env_snapshots", "__pycache__",
    "judge_metric_validity_100", "anonymous_eval_submission",
}
FORBIDDEN_SUFFIXES = {".pyc", ".log", ".pid", ".state", ".exit", ".docx", ".odt", ".html"}
ALLOWED_JSONL = {Path("aesthetic_eval_score_framework/tests/fixtures/synthetic_scores.jsonl")}
CLIP_LEGACY_ALLOWLIST = {
    Path("run.py"), Path("src/aesthetic_eval/metrics.py"), Path("README.md"),
    Path("docs/METRICS.md"), Path("results/v0.1.0/README.md"),
    Path("scripts/export_description_aggregates.py"),
    Path("scripts/export_random_baseline_summary.py"),
    Path("tests/test_release_results.py"),
    Path("scripts/check_release_integrity.py"),
}


def iter_files(root: Path):
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
        relatives = [Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw]
    else:
        relatives = [path.relative_to(root) for path in root.rglob("*") if path.is_file()]
    for relative in relatives:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            continue
        yield path, relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    total = 0
    personal_markers = [
        "/home/" + "Hu_xuanwei",
        "/data/students/" + "Hu_xuanwei",
        "/opt/" + "miniconda",
        "127." + "0.0.1",
    ]
    secret_patterns = [
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        re.compile(r"hf_[A-Za-z0-9]{24,}"),
        re.compile(r"(?i)(api[_-]?key|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    ]

    for path, relative in iter_files(root):
        total += path.stat().st_size
        if any(part in FORBIDDEN_DIRS or part.startswith("uniaa_outputs") or part.startswith("spice_cache") for part in relative.parts):
            errors.append(f"forbidden path: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden suffix: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"file exceeds 5 MiB: {relative}")
        if path.name == "predictions.jsonl" or path.name == "calc_accuracy.py":
            errors.append(f"forbidden release artifact: {relative}")
        if path.suffix == ".jsonl" and relative not in ALLOWED_JSONL:
            errors.append(f"non-synthetic JSONL is not public: {relative}")

        if path.suffix.lower() in {".pdf"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in personal_markers:
            if marker in text:
                errors.append(f"personal absolute path in {relative}")
        for pattern in secret_patterns:
            if pattern.search(text):
                errors.append(f"possible secret in {relative}")
        if "CLIPScore" in text and relative not in CLIP_LEGACY_ALLOWLIST:
            errors.append(f"legacy CLIPScore label outside compatibility documentation: {relative}")

    if total > 25 * 1024 * 1024:
        errors.append(f"repository payload exceeds 25 MiB: {total} bytes")
    if (root / "CITATION.cff").exists():
        errors.append("CITATION.cff must wait for confirmed paper metadata")

    env = {
        "AESTHETIC_DATA_ROOT": "/tmp/aesthetic-data",
        "AESTHETIC_MODEL_ROOT": "/tmp/aesthetic-models",
        "AESTHETIC_OUTPUT_ROOT": "/tmp/aesthetic-outputs",
        "AESTHETIC_METRIC_MODEL_ROOT": "/tmp/aesthetic-metrics",
        "AESTHETIC_JAVA": "/tmp/java",
        "ARTIMUSE_CONTEXT_FILE": "/tmp/artimuse-context.json",
        "UNIAA_1SHOT_CONTEXT_FILE": "/tmp/uniaa-context.json",
    }
    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        for yaml_path in sorted(root.rglob("*.yaml")):
            if ".git" in yaml_path.parts:
                continue
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            expanded = os.path.expandvars(str(data)).replace("${{ matrix.python-version }}", "3.10")
            if "${" in expanded:
                errors.append(f"unresolved config expression: {yaml_path.relative_to(root)}")
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    if errors:
        raise SystemExit("Release integrity failed:\n- " + "\n- ".join(sorted(set(errors))))
    print(f"Release integrity OK: {total} bytes across public files")


if __name__ == "__main__":
    main()
