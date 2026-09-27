#!/usr/bin/env python3
"""Build the fixed UNIAA-Describe 1-shot/500-sample protocol artifacts."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

FIXED_EXAMPLE_INDEX = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        actual = path.read_bytes()
        if actual != payload:
            raise RuntimeError(f"Refusing to overwrite drifted protocol artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--context-output", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    source_json = source_root / "UNIAA_Describe.json"
    target_dir = args.output_dir.expanduser().resolve()
    target_jsonl = target_dir / "test_500.jsonl"
    context_json = args.context_output.expanduser().resolve()
    manifest_json = target_dir / "manifest.json"

    rows = json.loads(source_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 501:
        raise RuntimeError(f"Expected exactly 501 source rows, got {len(rows) if isinstance(rows, list) else type(rows)}")

    required = {"img_path", "question", "correct_ans"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"Source row {index} lacks fields {sorted(missing)}")
        image_path = source_root / str(row["img_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing source image for row {index}: {image_path}")
        if not str(row["question"]).strip() or not str(row["correct_ans"]).strip():
            raise RuntimeError(f"Empty question/reference at source row {index}")

    source_sha256 = sha256(source_json)
    # Pre-registered positional rule: do not inspect reference content or compute
    # any statistic over the other evaluation references when choosing the example.
    example_index = FIXED_EXAMPLE_INDEX
    example = rows[example_index]
    test_indices = [index for index in range(len(rows)) if index != example_index]
    if len(test_indices) != 500:
        raise AssertionError("The held-out test split must contain exactly 500 rows")

    context = {
        "protocol_name": "uniaa_description_1shot_text_v1",
        "instruction": (
            "Describe and evaluate only the current image. Use the textual reference example solely "
            "to establish the requested analytical scope, tone, and level of detail. The example has "
            "no demonstration image and is not visual evidence for the current image. Do not mention, "
            "quote, or copy the example."
        ),
        "selection": {
            "source": str(source_json),
            "source_count": 501,
            "algorithm": "fixed_original_source_index_zero_based_0",
            "selection_scope": "source_order_only_without_reference_content_or_test_statistics",
            "methodological_boundary": "one_predeclared_source_row_used_as_text_only_demonstration_and_excluded_from_evaluation",
            "selected_original_index_zero_based": example_index,
            "test_policy": "exclude_the_selected_example_row_and_image_from_evaluation",
            "demonstration_modality": "text_only_no_demonstration_image",
        },
        "examples": [{
            "source_index_zero_based": example_index,
            "image": str(example["img_path"]),
            "question": str(example["question"]).strip(),
            "reference": str(example["correct_ans"]).strip(),
            "reference_sha256": hashlib.sha256(str(example["correct_ans"]).strip().encode("utf-8")).hexdigest(),
            "reference_word_count": len(str(example["correct_ans"]).split()),
        }],
    }

    derived_lines = []
    for original_index in test_indices:
        row = rows[original_index]
        derived = {
                "id": str(original_index),
                "image": str(row["img_path"]),
                "conversations": [
                    {"from": "human", "value": str(row["question"]).strip()},
                    {"from": "gpt", "value": str(row["correct_ans"]).strip()},
                ],
                "uniaa_original_index_zero_based": original_index,
                "source": row.get("source"),
                "dimension": row.get("dimension"),
                "concern": row.get("concern"),
        }
        derived_lines.append(json.dumps(derived, ensure_ascii=False) + "\n")
    derived_payload = "".join(derived_lines).encode("utf-8")
    context_payload = (json.dumps(context, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    # Existing artifacts are never overwritten. A rerun is a byte-exact audit.
    write_or_verify(target_jsonl, derived_payload)
    write_or_verify(context_json, context_payload)

    manifest = {
        "schema": "uniaa_description_1shot_500_v1",
        "source_json": str(source_json),
        "source_json_sha256": source_sha256,
        "source_count": 501,
        "source_unique_images": len({str(row["img_path"]) for row in rows}),
        "missing_images": 0,
        "selection_algorithm": context["selection"]["algorithm"],
        "selection_scope": context["selection"]["selection_scope"],
        "selection_methodological_boundary": context["selection"]["methodological_boundary"],
        "example_original_index_zero_based": example_index,
        "example_image": str(example["img_path"]),
        "example_image_sha256": sha256(source_root / str(example["img_path"])),
        "example_reference_sha256": context["examples"][0]["reference_sha256"],
        "test_count": len(test_indices),
        "test_original_indices_sha256": canonical_sha256(test_indices),
        "test_unique_images": len({str(rows[index]["img_path"]) for index in test_indices}),
        "example_image_in_test": str(example["img_path"]) in {str(rows[index]["img_path"]) for index in test_indices},
        "derived_jsonl": str(target_jsonl),
        "derived_jsonl_sha256": sha256(target_jsonl),
        "context_json": str(context_json),
        "context_json_sha256": sha256(context_json),
    }
    if manifest["source_unique_images"] != 501 or manifest["test_unique_images"] != 500 or manifest["example_image_in_test"]:
        raise RuntimeError(f"Split integrity failure: {manifest}")
    manifest_payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    write_or_verify(manifest_json, manifest_payload)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
