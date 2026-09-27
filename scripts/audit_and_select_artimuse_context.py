#!/usr/bin/env python3
"""Audit ArtiMuse annotations and select leakage-safe textual exemplars."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

DIM_RE = re.compile(r"(?:aspect|aespect) of\s+(.+?)[.\s]*$", re.IGNORECASE)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def fields(row):
    conversations = row.get("conversations", [])
    question = str(conversations[0].get("value", "")).strip()
    reference = str(conversations[1].get("value", "")).strip()
    match = DIM_RE.search(question)
    if not match:
        raise ValueError(f"Cannot parse dimension: {question}")
    return str(row.get("image", "")).strip(), question, reference, match.group(1).strip()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_set(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/Artimuse")
    parser.add_argument("--output", default="configs/prompt_styles/artimuse_16shot_text.json")
    parser.add_argument("--audit-output", default="artifacts/artimuse_audit.json")
    args = parser.parse_args()
    root = Path(args.root)
    test_rows = [(n, row, fields(row)) for n, row in read_jsonl(root / "text/test.jsonl")]
    train_rows = [(n, row, fields(row)) for n, row in read_jsonl(root / "text/train.jsonl")]
    score_rows = json.loads((root / "score/test.json").read_text(encoding="utf-8"))
    image_names = {path.name for path in (root / "images").iterdir() if path.is_file()}
    test_images = {item[2][0] for item in test_rows}
    test_answer_hashes = {sha256_text(item[2][2]) for item in test_rows}
    train_answer_hashes = {sha256_text(item[2][2]) for item in train_rows}

    candidates = defaultdict(list)
    for line_number, _, (image, question, reference, dimension) in train_rows:
        if image in test_images or sha256_text(reference) in test_answer_hashes:
            continue
        words = reference.split()
        if not 80 <= len(words) <= 180:
            continue
        candidates[dimension].append({
            "source_line": line_number, "image": image, "question": question,
            "reference": reference, "dimension": dimension, "word_count": len(words),
            "reference_sha256": sha256_text(reference), "tokens": token_set(reference),
        })

    selected = []
    for dimension in sorted(candidates):
        pool = candidates[dimension]
        target = median(item["word_count"] for item in pool)
        first = min(pool, key=lambda x: (abs(x["word_count"] - target), x["reference_sha256"]))
        remaining = [item for item in pool if item["image"] != first["image"]]
        second = min(remaining, key=lambda x: (jaccard(x["tokens"], first["tokens"]), abs(x["word_count"] - target), x["reference_sha256"]))
        for rank, item in enumerate((first, second), 1):
            clean = {key: value for key, value in item.items() if key != "tokens"}
            clean["selection_rank_within_dimension"] = rank
            clean["selection_reason"] = "closest_to_dimension_median_length" if rank == 1 else "minimum_lexical_overlap_with_first_then_nearest_median_length"
            selected.append(clean)

    dimensions = sorted(candidates)
    if len(dimensions) != 8 or len(selected) != 16 or Counter(x["dimension"] for x in selected) != Counter({d: 2 for d in dimensions}):
        raise RuntimeError("Expected exactly two examples for each of eight dimensions")
    payload = {
        "protocol_name": "artimuse_16shot_text_v1",
        "instruction": "Evaluate only the requested aesthetic dimension of the current image. Use the textual references below solely as examples of analytical scope, tone, and detail. They do not include demonstration images and must not be treated as visual evidence for the current image. Write one coherent evidence-based paragraph and do not mention the examples.",
        "selection": {"source": str((root / "text/train.jsonl").resolve()), "test_exclusion": "exclude every test image id and every exact test reference hash", "examples_per_dimension": 2, "word_count_filter": [80, 180], "ordering": "dimension_lexicographic_then_rank"},
        "examples": selected,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_counts = Counter(item[2][0] for item in test_rows)
    score_images = [str(item.get("image", "")) for item in score_rows]
    audit = {
        "text_train_sha256": sha256_file(root / "text/train.jsonl"),
        "text_test_sha256": sha256_file(root / "text/test.jsonl"),
        "score_test_sha256": sha256_file(root / "score/test.json"),
        "context_file_sha256": sha256_file(Path(args.output)),
        "image_file_count": len(image_names), "text_test_row_count": len(test_rows),
        "text_test_unique_images": len(text_counts),
        "text_test_dimension_counts": dict(sorted(Counter(item[2][3] for item in test_rows).items())),
        "text_test_rows_per_image": dict(sorted(Counter(text_counts.values()).items())),
        "text_test_missing_images": sorted(set(text_counts) - image_names),
        "score_test_row_count": len(score_rows), "score_test_unique_images": len(set(score_images)),
        "score_test_missing_images": sorted(set(score_images) - image_names),
        "score_without_text": sorted(set(score_images) - set(text_counts)),
        "test_images_present_in_train": len(test_images & {item[2][0] for item in train_rows}),
        "test_exact_rows_present_in_train": sum(sha256_text(x[2][2]) in train_answer_hashes for x in test_rows),
        "selected_context_count": len(selected),
        "selected_context_reference_hashes": [item["reference_sha256"] for item in selected],
    }
    Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_output).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
