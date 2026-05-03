import os
import platform
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from aesthetic_score_eval.adapters.base import BaseScoreAdapter
from aesthetic_score_eval.data import ScoreSample
from aesthetic_score_eval.io_utils import append_jsonl, utc_now_iso, write_json


def _chunked(items: List[ScoreSample], batch_size: int) -> List[List[ScoreSample]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _runtime_snapshot() -> Dict[str, str]:
    snap = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_count": str(torch.cuda.device_count() if torch.cuda.is_available() else 0),
    }
    if torch.cuda.is_available():
        snap["cuda_name_0"] = torch.cuda.get_device_name(0)
    return snap


def configure_runtime(base_cfg: dict) -> None:
    seed = int(base_cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    runtime = base_cfg.get("runtime", {})
    if bool(runtime.get("cudnn_benchmark", True)):
        torch.backends.cudnn.benchmark = True
    if bool(runtime.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True


def _normalize_score(raw_score: float, src_min: float, src_max: float, dst_min: float = 0.0, dst_max: float = 10.0) -> float:
    if src_max <= src_min:
        return float(max(dst_min, min(dst_max, raw_score)))
    x = (raw_score - src_min) / (src_max - src_min)
    y = dst_min + x * (dst_max - dst_min)
    return float(max(dst_min, min(dst_max, y)))


def _normalization_meta(src_min: float, src_max: float, dst_min: float, dst_max: float) -> Dict[str, object]:
    return {
        "raw_score_range": [src_min, src_max],
        "target_range": [dst_min, dst_max],
        "mapping": "linear_clip",
    }


def run_score_inference(
    adapter: BaseScoreAdapter,
    samples: List[ScoreSample],
    output_dir: str,
    base_cfg: dict,
    model_cfg: dict,
    dataset_meta: Dict[str, str],
) -> Dict[str, str]:
    batch_size = int(base_cfg.get("dataloader", {}).get("batch_size", 4))
    log_batch_time = bool(base_cfg.get("runtime", {}).get("log_batch_time", True))

    score_cfg = base_cfg.get("score", {})
    target_range = score_cfg.get("target_range", [0.0, 10.0])
    dst_min = float(target_range[0])
    dst_max = float(target_range[1])

    raw_range = model_cfg.get("raw_score_range", [0.0, 10.0])
    src_min = float(raw_range[0])
    src_max = float(raw_range[1])

    score_source = str(model_cfg.get("score_source", "unknown"))
    score_method = str(model_cfg.get("score_method", score_source))
    official_alignment = str(model_cfg.get("official_alignment", "unknown"))
    adapter_version = str(model_cfg.get("adapter_version", "v1"))
    model_name = str(model_cfg.get("model_name", "unknown"))
    task_name = str(base_cfg.get("task", "score"))
    norm_meta = _normalization_meta(src_min, src_max, dst_min, dst_max)

    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    run_meta_path = os.path.join(output_dir, "run_meta.json")
    if os.path.exists(predictions_path):
        os.remove(predictions_path)

    batches = _chunked(samples, batch_size)
    started = time.perf_counter()
    total_written = 0
    total_error = 0

    for idx, batch in enumerate(tqdm(batches, desc="infer-score")):
        t0 = time.perf_counter()
        scored = adapter.score_batch(batch)
        t1 = time.perf_counter()

        rows = []
        for sample, out in zip(batch, scored):
            raw_score = out.get("raw_score")
            err = out.get("error", "")
            parse_status = str(out.get("parse_status", "ok" if raw_score is not None and not err else "error"))
            score_0_10 = None
            if raw_score is not None and not err and parse_status == "ok":
                score_0_10 = _normalize_score(float(raw_score), src_min=src_min, src_max=src_max, dst_min=dst_min, dst_max=dst_max)
            else:
                total_error += 1

            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "image": sample.image,
                    "image_resolved": sample.image_resolved,
                    "gt_score": sample.gt_score,
                    "raw_score": raw_score,
                    "score_0_10": score_0_10,
                    "raw_response": out.get("raw_response", ""),
                    "parse_status": parse_status,
                    "score_source": score_source,
                    "score_method": str(out.get("score_method", score_method)),
                    "adapter_version": str(out.get("adapter_version", adapter_version)),
                    "official_alignment": str(out.get("official_alignment", official_alignment)),
                    "normalization": norm_meta,
                    "model": model_name,
                    "task": task_name,
                    "error": err,
                    "timestamp_utc": utc_now_iso(),
                }
            )

        append_jsonl(predictions_path, rows)
        total_written += len(rows)

        if log_batch_time:
            print(f"batch={idx} score_time={t1 - t0:.3f}s written={len(rows)} errors={sum(1 for r in rows if r['error'])}")

    elapsed = time.perf_counter() - started
    meta = {
        "model_name": model_name,
        "task": task_name,
        "sample_count": total_written,
        "error_count": total_error,
        "score_method": score_method,
        "official_alignment": official_alignment,
        "adapter_version": adapter_version,
        "normalization": norm_meta,
        "elapsed_seconds": elapsed,
        "dataset_meta": dataset_meta,
        "base_config": base_cfg,
        "model_config": model_cfg,
        "runtime_snapshot": _runtime_snapshot(),
    }
    write_json(run_meta_path, meta)

    return {
        "predictions": predictions_path,
        "run_meta": run_meta_path,
    }
