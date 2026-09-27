#!/usr/bin/env python3
"""Strict five-seed, zero-fixed-point mismatch baseline for UNIAA-Describe."""

import hashlib
import json
import os
import argparse
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026, 3407, 9999)
RAW_METRICS = (
    "BLEU", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
    "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR",
    "BERT-P", "BERT-R", "BERT-F1", "SBERT-Cos", "SPICE", "CLIP-Cos",
)

DATA_ROOT: Path
ANNOTATION: Path
OUTPUT_ROOT: Path
METRICS_CONFIG: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sattolo(size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    permutation = list(range(size))
    for index in range(size - 1, 0, -1):
        other = rng.randrange(index)
        permutation[index], permutation[other] = permutation[other], permutation[index]
    if sorted(permutation) != list(range(size)):
        raise RuntimeError(f"Invalid permutation for seed={seed}")
    if any(target == source for target, source in enumerate(permutation)):
        raise RuntimeError(f"Fixed point detected for seed={seed}")
    return permutation


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_inputs() -> dict[int, Path]:
    rows = json.loads(ANNOTATION.read_text(encoding="utf-8"))
    if len(rows) != 501:
        raise RuntimeError(f"Expected 501 UNIAA-Describe rows, got {len(rows)}")

    image_paths = [(DATA_ROOT / str(row["img_path"])).resolve() for row in rows]
    references = [str(row["correct_ans"]).strip() for row in rows]
    if len(set(image_paths)) != len(rows):
        raise RuntimeError("UNIAA-Describe image paths are not unique")
    if len(set(references)) != len(rows):
        raise RuntimeError("UNIAA-Describe references are not unique")
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing UNIAA-Describe images: {missing[:5]}")
    for path in image_paths:
        with Image.open(path) as image:
            image.load()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    inputs = {}
    manifests = {}
    for seed in SEEDS:
        permutation = sattolo(len(rows), seed)
        pred_file = OUTPUT_ROOT / f"predictions_mismatch_seed_{seed}.jsonl"
        with pred_file.open("w", encoding="utf-8") as handle:
            for target, source in enumerate(permutation):
                item = {
                    "sample_id": str(target),
                    "model": "reference_mismatch_sattolo",
                    "task": "description",
                    "image_resolved": str(image_paths[target]),
                    "question": str(rows[target]["question"]),
                    "prediction": references[source],
                    "reference": references[target],
                    "dimension": "description",
                    "mismatch_source_sample_id": str(source),
                    "mismatch_seed": seed,
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        permutation_hash = hashlib.sha256(
            json.dumps(permutation, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifests[str(seed)] = {
            "prediction_file": str(pred_file),
            "permutation_sha256": permutation_hash,
            "fixed_points": 0,
            "unique_sources": len(set(permutation)),
        }
        inputs[seed] = pred_file

    atomic_json(OUTPUT_ROOT / "manifest.json", {
        "protocol": "uniaa_description_global_sattolo_5seed_v1",
        "N": len(rows),
        "seeds": list(SEEDS),
        "annotation": str(ANNOTATION),
        "annotation_sha256": sha256_file(ANNOTATION),
        "image_count": len(image_paths),
        "missing_images": 0,
        "corrupt_images": 0,
        "reference_count": len(references),
        "unique_references": len(set(references)),
        "mismatch_constraint": "target_index != source_index for every sample",
        "length_penalty_reporting": "raw_metrics_only; shuffled predictions preserve the reference length marginal exactly",
        "seeds_manifest": manifests,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return inputs


def run_metrics(inputs: dict[int, Path]) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["TOKENIZERS_PARALLELISM"] = "false"
    for seed, pred_file in inputs.items():
        output_file = OUTPUT_ROOT / f"metrics_seed_{seed}.json"
        if output_file.is_file() and output_file.stat().st_size > 0:
            continue
        command = [
            sys.executable, str(ROOT / "run.py"), "eval",
            "--pred-file", str(pred_file),
            "--output-file", str(output_file),
            "--metrics-config", str(METRICS_CONFIG),
        ]
        print(f"[UNIAA MISMATCH] START seed={seed}", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        if not output_file.is_file() or output_file.stat().st_size == 0:
            raise RuntimeError(f"Metrics output missing for seed={seed}")
        print(f"[UNIAA MISMATCH] DONE seed={seed}", flush=True)


def aggregate() -> None:
    payloads = {
        seed: json.loads((OUTPUT_ROOT / f"metrics_seed_{seed}.json").read_text(encoding="utf-8"))
        for seed in SEEDS
    }
    summary = {
        "protocol": "uniaa_description_global_sattolo_5seed_v1",
        "N_per_seed": 501,
        "seeds": list(SEEDS),
        "fixed_points_per_seed": {str(seed): 0 for seed in SEEDS},
        "aggregation": "arithmetic_mean_and_sample_standard_deviation_across_five_seeds",
        "reported_metrics": "raw_only",
        "metrics": {},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for metric in RAW_METRICS:
        values = [float(payloads[seed]["metrics"][metric]) for seed in SEEDS]
        summary["metrics"][metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values),
            "values_by_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
        }
    atomic_json(OUTPUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    global DATA_ROOT, ANNOTATION, OUTPUT_ROOT, METRICS_CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--metrics-config", type=Path,
        default=ROOT / "configs/metrics/uniaa_description_v2.yaml",
    )
    args = parser.parse_args()
    DATA_ROOT = args.data_root.expanduser().resolve()
    ANNOTATION = DATA_ROOT / "UNIAA_Describe.json"
    OUTPUT_ROOT = args.output_dir.expanduser().resolve()
    METRICS_CONFIG = args.metrics_config.expanduser().resolve()
    inputs = prepare_inputs()
    run_metrics(inputs)
    aggregate()


if __name__ == "__main__":
    main()
