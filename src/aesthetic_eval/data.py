import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class EvalSample:
    sample_id: str
    image: str
    image_resolved: str
    question: str
    reference: str


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _extract_conversation_fields(conversations: Iterable[dict]) -> Tuple[str, str]:
    question = ""
    answer = ""
    for turn in conversations:
        role = str(turn.get("from", turn.get("role", ""))).lower().strip()
        value = str(turn.get("value", turn.get("content", ""))).strip()
        if role == "human" and not question:
            question = value
        elif role == "gpt" and not answer:
            answer = value
        elif role == "user" and not question:
            question = value
        elif role == "assistant" and not answer:
            answer = value
    return question, answer


def resolve_image_path(image: str, image_root: str, image_alt_root: Optional[str]) -> str:
    if os.path.isabs(image) and os.path.exists(image):
        return image

    p1 = os.path.join(image_root, image)
    if os.path.exists(p1):
        return p1

    if image_alt_root:
        p2 = os.path.join(image_alt_root, image)
        if os.path.exists(p2):
            return p2

    raise FileNotFoundError(f"Image not found: {image}")


def load_eval_samples(
    dataset_json: str,
    image_root: str,
    image_alt_root: Optional[str] = None,
    sample_limit: Optional[int] = None,
    strip_image_token: bool = True,
) -> Tuple[List[EvalSample], Dict[str, str]]:
    with open(dataset_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Dataset must be a list, got {type(raw)}")

    if sample_limit is not None:
        raw = raw[: sample_limit]

    samples: List[EvalSample] = []
    missing = 0
    for i, item in enumerate(raw):
        image = str(item.get("image", "")).strip()
        sid = str(item.get("id", i))
        conversations = item.get("conversations", [])
        q, a = _extract_conversation_fields(conversations)

        if strip_image_token:
            q = q.replace("<image>", "").strip()

        try:
            resolved = resolve_image_path(image, image_root, image_alt_root)
        except FileNotFoundError:
            missing += 1
            continue

        samples.append(
            EvalSample(
                sample_id=sid,
                image=image,
                image_resolved=resolved,
                question=q,
                reference=a,
            )
        )

    meta = {
        "dataset_json": str(Path(dataset_json).resolve()),
        "dataset_sha256": compute_file_sha256(dataset_json),
        "raw_count": str(len(raw)),
        "kept_count": str(len(samples)),
        "missing_image_count": str(missing),
    }
    return samples, meta


def load_and_resize_rgb(path: str, image_size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((image_size, image_size), resample=Image.BICUBIC)
