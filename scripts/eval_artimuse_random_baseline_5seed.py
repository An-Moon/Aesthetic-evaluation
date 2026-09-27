#!/usr/bin/env python3
import gc
import hashlib
import json
import os
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from bert_score import BERTScorer
from nltk.translate.meteor_score import meteor_score
from PIL import Image
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from aesthetic_eval.spice_metric import compute_spice_strict


ROOT = Path(__file__).resolve().parents[1]
PRED_FILE = Path(os.environ["ARTIMUSE_REFERENCE_PRED_FILE"]).expanduser().resolve()
OUTPUT_DIR = Path(os.environ["ARTIMUSE_BASELINE_OUTPUT_DIR"]).expanduser().resolve()
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
SEEDS = [42, 123, 2026, 3407, 9999]
DEVICE = os.environ.get("AESTHETIC_BASELINE_DEVICE", "cuda")
METRIC_ROOT = Path(os.environ["AESTHETIC_METRIC_MODEL_ROOT"]).expanduser().resolve()
JAVA_PATH = Path(os.environ["AESTHETIC_JAVA"]).expanduser().resolve()


def sattolo(indices, rng):
    result = list(indices)
    for index in range(len(result) - 1, 0, -1):
        other = rng.randrange(index)
        result[index], result[other] = result[other], result[index]
    if len(result) > 1 and any(left == right for left, right in zip(indices, result)):
        raise RuntimeError("Sattolo permutation unexpectedly contains a fixed point")
    return result


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def seed_path(seed):
    return OUTPUT_DIR / f"seed_{seed}.json"


def load_seed(seed, permutation):
    path = seed_path(seed)
    expected_hash = hashlib.sha256(np.asarray(permutation, dtype=np.int64).tobytes()).hexdigest()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("permutation_sha256") != expected_hash:
            raise RuntimeError(f"Permutation identity mismatch: {path}")
        return payload
    payload = {
        "protocol": "artimuse_dimension_matched_sattolo_5seed_v1",
        "seed": seed,
        "N": 7984,
        "fixed_points": 0,
        "permutation_sha256": expected_hash,
        "source_pred_file": str(PRED_FILE.resolve()),
        "metrics": {},
    }
    atomic_json(path, payload)
    return payload


def save_seed(payload):
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(seed_path(payload["seed"]), payload)


def feature_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "pooler_output"):
        return value.pooler_output
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    raise TypeError(f"Unsupported feature output type: {type(value)!r}")


def main():
    rows = [json.loads(line) for line in PRED_FILE.open(encoding="utf-8") if line.strip()]
    if len(rows) != 7984:
        raise RuntimeError(f"Expected 7984 rows, got {len(rows)}")
    refs = [str(row["reference"]) for row in rows]
    dimensions = [str(row["dimension"]) for row in rows]
    images = [str(row["image_resolved"]) for row in rows]
    groups = defaultdict(list)
    for index, dimension in enumerate(dimensions):
        groups[dimension].append(index)
    if sorted(len(indices) for indices in groups.values()) != [998] * 8:
        raise RuntimeError("Expected eight dimensions with 998 rows each")

    permutations = {}
    payloads = {}
    for seed in SEEDS:
        rng = random.Random(seed)
        permutation = [-1] * len(rows)
        for indices in groups.values():
            for target, source in zip(indices, sattolo(indices, rng)):
                permutation[target] = source
        if any(index < 0 for index in permutation):
            raise RuntimeError(f"Incomplete permutation for seed={seed}")
        if any(target == source for target, source in enumerate(permutation)):
            raise RuntimeError(f"Fixed point detected for seed={seed}")
        if any(dimensions[target] != dimensions[source] for target, source in enumerate(permutation)):
            raise RuntimeError(f"Cross-dimension pair detected for seed={seed}")
        permutations[seed] = permutation
        payloads[seed] = load_seed(seed, permutation)

    lexical = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for seed in SEEDS:
        payload = payloads[seed]
        needed = {"BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"}
        if needed.issubset(payload["metrics"]):
            continue
        preds = [refs[source] for source in permutations[seed]]
        rouge1, rouge2, rougel, meteor = [], [], [], []
        for pred, ref in zip(preds, refs):
            scores = lexical.score(ref, pred)
            rouge1.append(float(scores["rouge1"].fmeasure))
            rouge2.append(float(scores["rouge2"].fmeasure))
            rougel.append(float(scores["rougeL"].fmeasure))
            meteor.append(float(meteor_score([ref.split()], pred.split())))
        payload["metrics"].update({
            "BLEU": float(BLEU(effective_order=True).corpus_score(preds, [refs]).score / 100.0),
            "ROUGE-1": statistics.fmean(rouge1),
            "ROUGE-2": statistics.fmean(rouge2),
            "ROUGE-L": statistics.fmean(rougel),
            "METEOR": statistics.fmean(meteor),
        })
        save_seed(payload)
        print(f"[STRICT RANDOM] lexical complete seed={seed}", flush=True)

    sbert = SentenceTransformer(
        str(METRIC_ROOT / "all-mpnet-base-v2_e8c3b32"), device="cpu"
    )
    ref_sbert = sbert.encode(
        refs, batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True
    )
    for seed in SEEDS:
        payload = payloads[seed]
        if "SBERT-Cos" not in payload["metrics"]:
            permutation = np.asarray(permutations[seed], dtype=np.int64)
            payload["metrics"]["SBERT-Cos"] = float(
                np.mean(np.sum(ref_sbert * ref_sbert[permutation], axis=1))
            )
            save_seed(payload)
    del sbert, ref_sbert
    gc.collect()
    print("[STRICT RANDOM] SBERT complete", flush=True)

    clip_path = str(METRIC_ROOT / "clip-vit-base-patch32_3d74acf")
    clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(DEVICE).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
    max_text_len = int(clip_model.config.text_config.max_position_embeddings)
    text_features = []
    with torch.no_grad():
        for start in range(0, len(refs), 64):
            inputs = clip_processor(
                text=refs[start:start + 64], return_tensors="pt", padding=True,
                truncation=True, max_length=max_text_len,
            )
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
            features = feature_tensor(clip_model.get_text_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            text_features.append(features.cpu())
    text_features = torch.cat(text_features).numpy()

    unique_images = list(dict.fromkeys(images))
    image_features_by_path = {}
    with torch.no_grad():
        for start in range(0, len(unique_images), 64):
            paths = unique_images[start:start + 64]
            opened = [Image.open(path).convert("RGB") for path in paths]
            inputs = clip_processor(images=opened, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(DEVICE)
            features = feature_tensor(clip_model.get_image_features(pixel_values=pixel_values))
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            for path, feature in zip(paths, features.cpu().numpy()):
                image_features_by_path[path] = feature
    image_features = np.stack([image_features_by_path[path] for path in images])
    for seed in SEEDS:
        payload = payloads[seed]
        if "CLIP-Cos" not in payload["metrics"]:
            permutation = np.asarray(permutations[seed], dtype=np.int64)
            payload["metrics"]["CLIP-Cos"] = float(
                np.mean(np.sum(text_features[permutation] * image_features, axis=1))
            )
            save_seed(payload)
    del clip_model, clip_processor, text_features, image_features, image_features_by_path
    gc.collect()
    torch.cuda.empty_cache()
    print("[STRICT RANDOM] CLIP complete", flush=True)

    bert = BERTScorer(
        model_type=str(METRIC_ROOT / "deberta-base-mnli_a80a6eb"),
        num_layers=9, batch_size=8, device=DEVICE, use_fast_tokenizer=False,
    )
    if getattr(bert._tokenizer, "model_max_length", 512) > 1_000_000:
        bert._tokenizer.model_max_length = 512
    for seed in SEEDS:
        payload = payloads[seed]
        if "BERT-F1" not in payload["metrics"]:
            preds = [refs[source] for source in permutations[seed]]
            precision, recall, f1 = bert.score(preds, refs)
            payload["metrics"].update({
                "BERT-P": float(precision.mean().item()),
                "BERT-R": float(recall.mean().item()),
                "BERT-F1": float(f1.mean().item()),
            })
        save_seed(payload)
        print(f"[STRICT RANDOM] BERT complete seed={seed}", flush=True)
    del bert
    gc.collect()
    torch.cuda.empty_cache()

    for seed in SEEDS:
        payload = payloads[seed]
        if "SPICE" not in payload["metrics"]:
            preds = [refs[source] for source in permutations[seed]]
            score, per_sample = compute_spice_strict(
                preds, refs,
                java_path=str(JAVA_PATH),
                cache_dir=str(OUTPUT_DIR / "spice_cache_disabled"),
                timeout_seconds=2400, chunk_size=500, cache_enabled=False,
                threads=8, java_heap_gb=24, checkpoint_enabled=True,
                checkpoint_dir=(
                    OUTPUT_DIR / "spice_checkpoints" / f"seed_{seed}"
                ),
            )
            if len(per_sample) != len(rows):
                raise RuntimeError(f"SPICE coverage mismatch seed={seed}: {len(per_sample)}")
            payload["metrics"]["SPICE"] = float(score)
        payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_seed(payload)
        print(f"[STRICT RANDOM] SPICE complete seed={seed}", flush=True)

    metric_names = [
        "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR",
        "BERT-P", "BERT-R", "BERT-F1", "SBERT-Cos", "SPICE", "CLIP-Cos",
    ]
    summary = {
        "protocol": "artimuse_dimension_matched_sattolo_5seed_v1",
        "seeds": SEEDS,
        "N_per_seed": len(rows),
        "fixed_points_per_seed": {str(seed): 0 for seed in SEEDS},
        "aggregation": "arithmetic_mean_and_sample_standard_deviation_across_five_seeds",
        "metrics": {},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for name in metric_names:
        values = [float(payloads[seed]["metrics"][name]) for seed in SEEDS]
        summary["metrics"][name] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values),
            "values_by_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
        }
    atomic_json(SUMMARY_FILE, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
