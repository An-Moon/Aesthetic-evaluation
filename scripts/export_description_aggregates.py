#!/usr/bin/env python3
"""Export aggregate-only Description artifacts from a private experiment tree.

The exporter deliberately omits prompts, references, image paths, predictions,
and per-sample metric details. It is intended for release preparation, not for
running evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


RUNS = {
    "ArtiMuse": {
        "expected_n": 7984,
        "protocol": "artimuse_description_0shot_v1",
        "models": {
            "InternVL3.5-8B": ("General MLLMs", "outputs/artimuse/internvl_description_20260819_131211/metrics_v2.json"),
            "GLM-4.6V-Flash": ("General MLLMs", "outputs/artimuse/glm4v_flash_description_20260828_151146_control_clean_v1/metrics_v2.json"),
            "Qwen3.5-9B": ("General MLLMs", "outputs/artimuse/qwen3_5_9b_description_20260829_005424/metrics_v2.json"),
            "Qwen3-VL-8B": ("General MLLMs", "outputs/artimuse/qwen3_vl_description_20260828_151127/metrics_v2.json"),
            "LLaVA-OneVision-8B": ("General MLLMs", "outputs/artimuse/llava_onevision_description_20260828_231033/metrics_v2.json"),
            "UNIAA": ("IAA-specific MLLMs", "outputs/artimuse/uniaa_description_20260827_133829/metrics_v2.json"),
            "UniPercept": ("IAA-specific MLLMs", "outputs/artimuse/unipercept_description_20260828_183108/metrics_v2.json"),
            "Aes-R1": ("IAA-specific MLLMs", "outputs/artimuse/aes_r1_description_20260829_151006/metrics_v2.json"),
            "ArtQuant": ("IAA-specific MLLMs", "outputs/artimuse/artquant_description_20260819_064735/metrics_v2.json"),
        },
    },
    "UNIAA": {
        "expected_n": 500,
        "protocol": "uniaa_description_fixed_first_1shot_greedy_v2",
        "models": {
            "InternVL3.5-8B": ("General MLLMs", "outputs/uniaa_description_1shot_greedy_v2_models/internvl/internvl_description_20260830_135951/metrics_uniaa_v2.json"),
            "GLM-4.6V-Flash": ("General MLLMs", "outputs/uniaa_description_1shot_greedy_v2_models/glm4v_flash/glm4v_flash_description_20260830_161600/metrics_uniaa_v2.json"),
            "Qwen3.5-9B": ("General MLLMs", "outputs/uniaa_description_1shot_qwen35_greedy_v2/qwen3_5_9b_description_20260829_214112/metrics_uniaa_v2.json"),
            "Qwen3-VL-8B": ("General MLLMs", "outputs/uniaa_description_1shot_greedy_v2_models/qwen3_vl/qwen3_vl_description_20260830_002438/metrics_uniaa_v2.json"),
            "LLaVA-OneVision-8B": ("General MLLMs", "outputs/uniaa_description_1shot_greedy_v2_models/llava_onevision/llava_onevision_description_20260830_013629/metrics_uniaa_v2.json"),
            "UNIAA": ("IAA-specific MLLMs", "outputs/uniaa_description_1shot_greedy_v2/uniaa_description_20260829_214231/metrics_uniaa_v2.json"),
            "UniPercept": ("IAA-specific MLLMs", "outputs/uniaa_description_1shot_greedy_v2_models/unipercept/unipercept_description_20260830_201043/metrics_uniaa_v2.json"),
            "Aes-R1": ("IAA-specific MLLMs", "outputs/uniaa_description_1shot_greedy_v2_models/aes_r1/aes_r1_description_20260830_213317/metrics_uniaa_v2.json"),
            "ArtQuant": ("IAA-specific MLLMs", "outputs/uniaa_description_1shot_greedy_v2_models/artquant/artquant_description_20260830_151222/metrics_uniaa_v2.json"),
        },
    },
}

METRICS = ("Pred-Len", "BLEU", "ROUGE-L", "METEOR", "BERT-F1", "SBERT-Cos", "SPICE", "CLIP-Cos")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_metrics(values: dict) -> dict:
    values = dict(values)
    if "CLIP-Cos" not in values and "CLIPScore" in values:
        values["CLIP-Cos"] = values["CLIPScore"]
    missing = [key for key in METRICS if key not in values]
    if missing:
        raise KeyError(f"Missing aggregate metric(s): {missing}")
    return values


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    overall: list[dict] = []
    dimensions: list[dict] = []
    provenance: list[dict] = []
    for dataset, spec in RUNS.items():
        for model, (group, relative) in spec["models"].items():
            source = args.source_repo / relative
            payload = json.loads(source.read_text(encoding="utf-8"))
            values = normalize_metrics(payload["metrics"])
            if int(values.get("N", payload["metrics"].get("N", -1))) != spec["expected_n"]:
                raise ValueError(f"Unexpected N in {source}: {payload['metrics'].get('N')}")
            overall.append({
                "dataset": dataset,
                "protocol": spec["protocol"],
                "group": group,
                "model": model,
                "N": spec["expected_n"],
                **{key: values[key] for key in METRICS},
                "validity_status": "validated",
                "primary_analysis": "true",
            })
            pred_file = Path(str(payload.get("pred_file", "")))
            provenance.append({
                "dataset": dataset,
                "model": model,
                "metric_artifact_sha256": sha256(source),
                "prediction_sha256": sha256(pred_file) if pred_file.is_file() else "not_recorded",
            })
            if dataset == "ArtiMuse":
                by_dimension = payload.get("metrics_by_dimension", {})
                if len(by_dimension) != 8:
                    raise ValueError(f"Expected 8 dimensions in {source}, found {len(by_dimension)}")
                for dimension, raw_dimension in sorted(by_dimension.items()):
                    dvalues = normalize_metrics(raw_dimension)
                    dimensions.append({
                        "dataset": dataset,
                        "group": group,
                        "model": model,
                        "dimension": dimension,
                        "N": int(raw_dimension["N"]),
                        **{key: dvalues[key] for key in METRICS},
                    })

    metric_fields = list(METRICS)
    write_csv(
        args.output_dir / "description_overall.csv",
        overall,
        ["dataset", "protocol", "group", "model", "N", *metric_fields, "validity_status", "primary_analysis"],
    )
    write_csv(
        args.output_dir / "artimuse_description_by_dimension.csv",
        dimensions,
        ["dataset", "group", "model", "dimension", "N", *metric_fields],
    )
    write_csv(
        args.output_dir / "description_provenance.csv",
        provenance,
        ["dataset", "model", "metric_artifact_sha256", "prediction_sha256"],
    )


if __name__ == "__main__":
    main()
