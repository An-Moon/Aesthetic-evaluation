import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ScoreSample:
    sample_id: str
    image: str
    image_resolved: str
    gt_score: float
    dataset: str
    split: str


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_image_path(image: str, image_root: str, image_alt_root: Optional[str]) -> str:
    if os.path.isabs(image) and os.path.exists(image):
        return image

    if image_root:
        p1 = os.path.join(image_root, image)
        if os.path.exists(p1):
            return p1

    if image_alt_root:
        p2 = os.path.join(image_alt_root, image)
        if os.path.exists(p2):
            return p2

    raise FileNotFoundError(f"Image not found: {image}")


def _load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sample_rows(rows: List[dict], sample_limit: Optional[int], sample_size: Optional[int], seed: int, shuffle: bool) -> List[dict]:
    out = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(out)

    if sample_size is not None:
        out = out[: max(0, min(sample_size, len(out)))]
    if sample_limit is not None:
        out = out[: max(0, min(sample_limit, len(out)))]
    return out


def load_score_samples(
    dataset_jsonl: str,
    image_root: str,
    image_alt_root: Optional[str] = None,
    sample_limit: Optional[int] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    shuffle: bool = True,
    sample_manifest_out: Optional[str] = None,
    sample_manifest_in: Optional[str] = None,
) -> Tuple[List[ScoreSample], Dict[str, str]]:
    raw = _load_jsonl(dataset_jsonl)
    manifest_used = ""
    if sample_manifest_in:
        with open(sample_manifest_in, "r", encoding="utf-8") as f:
            ordered_ids = [line.strip() for line in f if line.strip()]
        id_set = set(ordered_ids)
        id_to_row: Dict[str, dict] = {}
        for i, item in enumerate(raw):
            sid = str(item.get("sample_id", item.get("id", i)))
            if sid in id_set and sid not in id_to_row:
                id_to_row[sid] = item
        raw = [id_to_row[sid] for sid in ordered_ids if sid in id_to_row]
        manifest_used = str(Path(sample_manifest_in).resolve())
    else:
        raw = _sample_rows(raw, sample_limit=sample_limit, sample_size=sample_size, seed=seed, shuffle=shuffle)

    samples: List[ScoreSample] = []
    missing = 0
    skipped = 0
    ids_for_manifest: List[str] = []

    for i, item in enumerate(raw):
        image = str(item.get("image", "")).strip()
        if not image:
            skipped += 1
            continue

        if "gt_score" not in item:
            skipped += 1
            continue

        try:
            gt_score = float(item.get("gt_score"))
        except Exception:
            skipped += 1
            continue

        sid = str(item.get("sample_id", item.get("id", i)))
        dataset = str(item.get("dataset", "unknown"))
        split = str(item.get("split", "unknown"))

        try:
            resolved = resolve_image_path(image, image_root, image_alt_root)
        except FileNotFoundError:
            missing += 1
            continue

        samples.append(
            ScoreSample(
                sample_id=sid,
                image=image,
                image_resolved=resolved,
                gt_score=gt_score,
                dataset=dataset,
                split=split,
            )
        )
        ids_for_manifest.append(sid)

    if sample_manifest_out:
        parent = os.path.dirname(os.path.abspath(sample_manifest_out))
        os.makedirs(parent, exist_ok=True)
        with open(sample_manifest_out, "w", encoding="utf-8") as f:
            for sid in ids_for_manifest:
                f.write(f"{sid}\n")

    meta = {
        "dataset_jsonl": str(Path(dataset_jsonl).resolve()),
        "dataset_sha256": compute_file_sha256(dataset_jsonl),
        "raw_count": str(len(raw)),
        "kept_count": str(len(samples)),
        "missing_image_count": str(missing),
        "skipped_count": str(skipped),
        "sample_manifest_in": manifest_used,
    }
    return samples, meta
