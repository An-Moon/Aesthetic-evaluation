#!/usr/bin/env python3
"""Create a provenance-preserving JSONL copy with model control tokens removed."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


CONTROL_TOKEN = re.compile(r"<\|[^<>\r\n]*?\|>")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--reuse-verified", action="store_true")
    args = parser.parse_args()

    if args.output.exists() or args.manifest.exists():
        if not (args.reuse_verified and args.output.is_file() and args.manifest.is_file()):
            raise FileExistsError("Refusing to overwrite an existing cleaned output or manifest")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        checks = {
            "input_sha256": sha256(args.input),
            "output_sha256": sha256(args.output),
            "rows": sum(1 for _ in args.output.open("r", encoding="utf-8")),
        }
        for key, actual in checks.items():
            if manifest.get(key) != actual:
                raise RuntimeError(
                    f"Existing clean artifact failed verification for {key}: "
                    f"manifest={manifest.get(key)!r}, actual={actual!r}"
                )
        if checks["rows"] != args.expected_rows:
            raise RuntimeError(
                f"Existing clean artifact coverage mismatch: {checks['rows']} != {args.expected_rows}"
            )
        print(json.dumps({"status": "reused_verified", **manifest}, ensure_ascii=False))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    changed_rows = removed_tokens = empty_after_cleaning = 0
    row_count = 0

    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "x", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line at {line_number}")
            row = json.loads(line)
            prediction = str(row.get("prediction", ""))
            matches = CONTROL_TOKEN.findall(prediction)
            cleaned = CONTROL_TOKEN.sub("", prediction).strip()
            if matches:
                changed_rows += 1
                removed_tokens += len(matches)
            if not cleaned:
                empty_after_cleaning += 1
            row["prediction"] = cleaned
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            row_count += 1

    if row_count != args.expected_rows:
        raise RuntimeError(f"Coverage mismatch: rows={row_count}, expected={args.expected_rows}")
    if changed_rows == 0:
        raise RuntimeError("No control tokens were found; refusing an unverified no-op clean")
    if empty_after_cleaning:
        raise RuntimeError(f"Cleaning produced {empty_after_cleaning} empty predictions")

    leftover_rows = 0
    with args.output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if CONTROL_TOKEN.search(str(json.loads(line)["prediction"])):
                leftover_rows += 1
    if leftover_rows:
        raise RuntimeError(f"Control tokens remain in {leftover_rows} rows")

    manifest = {
        "schema": "aesthetic_prediction_control_token_clean_v1",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "rows": row_count,
        "changed_rows": changed_rows,
        "removed_tokens": removed_tokens,
        "empty_after_cleaning": empty_after_cleaning,
        "leftover_control_token_rows": leftover_rows,
        "regex": CONTROL_TOKEN.pattern,
        "policy": "remove_angle_pipe_control_tokens_only_then_strip_outer_whitespace",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
