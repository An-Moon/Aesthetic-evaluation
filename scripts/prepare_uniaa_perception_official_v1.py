#!/usr/bin/env python3
"""Prepare a source-faithful UNIAA perception protocol and stratified smoke set."""

from __future__ import annotations

import hashlib
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path


LABELS = "ABCDE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(index: int, row: dict, source_root: Path) -> dict:
    required = {"img_path", "question", "candidates", "correct_ans", "dimension", "type"}
    missing = sorted(required - set(row))
    if missing:
        raise KeyError(f"row {index} missing fields: {missing}")
    candidates = row["candidates"]
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 5:
        raise ValueError(f"row {index} has invalid candidates: {candidates!r}")
    normalized = [str(value).strip() for value in candidates]
    if len(set(normalized)) != len(normalized) or any(not value for value in normalized):
        raise ValueError(f"row {index} has empty/duplicate candidates")
    correct = str(row["correct_ans"]).strip()
    if sum(correct == value for value in normalized) != 1:
        raise ValueError(f"row {index} correct answer is not a unique candidate: {correct!r}")
    image = str(row["img_path"]).strip()
    if not (source_root / image).is_file():
        raise FileNotFoundError(source_root / image)

    # This wording and A./B./... formatting exactly follow the official
    # aesthetic_perception.py English prompt construction.
    prompt = str(row["question"]).strip() + "\nChoose between one of the options as follows:\n"
    prompt += "".join(f"{LABELS[i]}. {answer}\n" for i, answer in enumerate(normalized))
    return {
        "id": str(index),
        "image": image,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": correct},
        ],
        "candidates": normalized,
        "correct_ans": correct,
        "correct_label": LABELS[normalized.index(correct)],
        "dimension": str(row["dimension"]).strip(),
        "question_type": str(row["type"]).strip(),
        "category": row.get("category"),
        "dataset": row.get("dataset"),
        "source": row.get("source"),
        "source_index": index,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    source = source_root / "UNIAA_QA.json"
    out = args.output_dir.expanduser().resolve()

    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 5354:
        raise RuntimeError(f"Expected 5354 source rows, got {type(raw)} / {len(raw)}")
    rows = [convert(index, row, source_root) for index, row in enumerate(raw)]
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Prepared sample IDs are not unique")

    # Deterministically take the first two rows from each dimension/type cell.
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(row["dimension"], row["question_type"])].append(row)
    smoke = []
    for key in sorted(cells):
        smoke.extend(cells[key][:2])
    smoke.sort(key=lambda row: int(row["id"]))

    out.mkdir(parents=True, exist_ok=True)
    full_path = out / "test_5354.jsonl"
    smoke_path = out / "smoke_stratified.jsonl"
    write_jsonl(full_path, rows)
    write_jsonl(smoke_path, smoke)
    manifest = {
        "protocol_name": "official_uniaa_perception_v1",
        "source_json": str(source),
        "source_sha256": sha256(source),
        "official_script": str(source_root.parent / "aesthetic_perception.py"),
        "full_jsonl": str(full_path),
        "full_sha256": sha256(full_path),
        "full_count": len(rows),
        "smoke_jsonl": str(smoke_path),
        "smoke_sha256": sha256(smoke_path),
        "smoke_count": len(smoke),
        "dimensions": dict(sorted(Counter(row["dimension"] for row in rows).items())),
        "question_types": dict(sorted(Counter(row["question_type"] for row in rows).items())),
        "candidate_counts": dict(sorted(Counter(len(row["candidates"]) for row in rows).items())),
        "smoke_selection": "first two source-order samples per observed (dimension, question_type) cell",
        "prompt": "official English question + Choose between one of the options as follows + A./B./...",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
