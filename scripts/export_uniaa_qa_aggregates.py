#!/usr/bin/env python3
"""Export aggregate UNIAA QA results and paired exact tests without details."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path


MODELS = {
    "Aes-R1": ("IAA-specific MLLMs", "aes_r1"),
    "ArtiMuse": ("IAA-specific MLLMs", "artimuse"),
    "GLM-4.6V-Flash": ("General MLLMs", "glm4v_flash"),
    "InternVL3.5-8B": ("General MLLMs", "internvl"),
    "LLaVA-OneVision-8B": ("General MLLMs", "llava_onevision"),
    "Qwen3.5-9B": ("General MLLMs", "qwen3_5_9b"),
    "Qwen3-VL-8B": ("General MLLMs", "qwen3_vl"),
    "UNIAA": ("IAA-specific MLLMs", "uniaa"),
    "UniPercept": ("IAA-specific MLLMs", "unipercept"),
}


def exact_two_sided_binomial(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payloads: dict[str, dict] = {}
    overall: list[dict] = []
    subgroup: list[dict] = []
    for model, (group, folder) in MODELS.items():
        path = args.qa_root / folder / "accuracy_strict_v2.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload["N"]) != 5354 or len(payload.get("details", [])) != 5354:
            raise ValueError(f"Incomplete QA artifact: {path}")
        ids = [str(row["sample_id"]) for row in payload["details"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs: {path}")
        payloads[model] = payload
        overall.append({
            "model": model,
            "group": group,
            "N": payload["N"],
            "correct": payload["correct"],
            "accuracy": payload["accuracy"],
            "valid": payload["valid"],
            "invalid": payload["invalid"],
            "valid_rate": payload["valid_rate"],
            "dataset_sha256": payload["dataset_sha256"],
            "prediction_sha256": payload["pred_sha256"],
            "parser_sha256": payload["parser_sha256"],
            "validity_status": "validated",
        })
        for kind, key in (("dimension", "by_dimension"), ("question_type", "by_question_type"), ("source", "by_dataset")):
            for name, values in sorted(payload[key].items()):
                subgroup.append({
                    "model": model,
                    "group": group,
                    "subgroup_type": kind,
                    "subgroup": name,
                    **{field: values[field] for field in ("N", "correct", "accuracy", "valid", "valid_rate")},
                })

    pair_rows: list[dict] = []
    for model_a, model_b in itertools.combinations(MODELS, 2):
        correctness = {}
        for model in (model_a, model_b):
            correctness[model] = {
                str(row["sample_id"]): bool(row["is_correct"])
                for row in payloads[model]["details"]
            }
        if set(correctness[model_a]) != set(correctness[model_b]):
            raise ValueError(f"Sample coverage differs: {model_a} vs {model_b}")
        a_only = sum(correctness[model_a][sid] and not correctness[model_b][sid] for sid in correctness[model_a])
        b_only = sum(correctness[model_b][sid] and not correctness[model_a][sid] for sid in correctness[model_a])
        pair_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "N": 5354,
            "a_correct_b_wrong": a_only,
            "a_wrong_b_correct": b_only,
            "discordant": a_only + b_only,
            "accuracy_delta_a_minus_b": payloads[model_a]["accuracy"] - payloads[model_b]["accuracy"],
            "p_exact": exact_two_sided_binomial(a_only, a_only + b_only),
        })
    adjusted = holm([float(row["p_exact"]) for row in pair_rows])
    for row, value in zip(pair_rows, adjusted):
        row["p_holm"] = value
        row["significant_0_05"] = str(value < 0.05).lower()

    write_csv(args.output_dir / "uniaa_qa_overall.csv", overall, list(overall[0]))
    write_csv(args.output_dir / "uniaa_qa_subgroups.csv", subgroup, list(subgroup[0]))
    write_csv(args.output_dir / "uniaa_qa_paired_significance.csv", pair_rows, list(pair_rows[0]))


if __name__ == "__main__":
    main()
