import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample
from aesthetic_eval.io_utils import append_jsonl, utc_now_iso, write_json


def _chunked(items: List[EvalSample], batch_size: int) -> List[List[EvalSample]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _read_completed_sample_ids(predictions_path: str) -> set:
    completed = set()
    if not os.path.exists(predictions_path):
        return completed
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed JSONL in resume predictions at line {exc.lineno}: {predictions_path}"
                ) from exc
            sid = row.get("sample_id")
            if sid is None or not str(sid).strip():
                raise RuntimeError(f"Missing sample_id in resume predictions: {predictions_path}")
            sid = str(sid)
            if sid in completed:
                raise RuntimeError(f"Duplicate sample_id={sid} in resume predictions: {predictions_path}")
            completed.add(sid)
    return completed


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
    runtime = base_cfg.get("runtime", {})
    deterministic = bool(runtime.get("deterministic", True))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    seed = int(base_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", False))
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime.get("allow_tf32", False))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(runtime.get("allow_tf32", False))
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def run_inference(
    adapter: BaseAdapter,
    samples: List[EvalSample],
    output_dir: str,
    model_name: str,
    task_name: str,
    base_cfg: dict,
    model_cfg: dict,
    dataset_meta: Dict[str, str],
    resume: bool = False,
) -> Dict[str, str]:
    batch_size = int(base_cfg.get("dataloader", {}).get("batch_size", 4))
    log_batch_time = bool(base_cfg.get("runtime", {}).get("log_batch_time", True))
    strict_inference = bool(base_cfg.get("runtime", {}).get("strict_inference", False))

    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    run_meta_path = os.path.join(output_dir, "run_meta.json")

    expected_sample_ids = [sample.sample_id for sample in samples]
    if len(expected_sample_ids) != len(set(expected_sample_ids)):
        raise RuntimeError("Input samples contain duplicate sample_id values")
    completed_sample_ids = _read_completed_sample_ids(predictions_path) if resume else set()
    unexpected_completed = completed_sample_ids - set(expected_sample_ids)
    if unexpected_completed:
        raise RuntimeError(
            f"Resume predictions contain {len(unexpected_completed)} IDs outside the current dataset"
        )
    if os.path.exists(predictions_path) and not resume:
        os.remove(predictions_path)

    if completed_sample_ids:
        samples = [s for s in samples if s.sample_id not in completed_sample_ids]

    batches = _chunked(samples, batch_size)
    started = time.perf_counter()
    initial_written = len(completed_sample_ids)
    new_written = 0
    total_written = initial_written

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        if batches:
            future = executor.submit(adapter.prepare_batch, batches[0])

        for idx in tqdm(range(len(batches)), desc="infer"):
            prep_start = time.perf_counter()
            prepared, valid_samples, prompts = future.result()
            prep_time = time.perf_counter() - prep_start
            if strict_inference and len(valid_samples) != len(batches[idx]):
                raise RuntimeError(f"Strict preprocessing coverage failed in batch {idx}: valid={len(valid_samples)} expected={len(batches[idx])}")

            if idx + 1 < len(batches):
                future = executor.submit(adapter.prepare_batch, batches[idx + 1])

            if prepared is None or not valid_samples:
                continue

            gen_start = time.perf_counter()
            outputs = adapter.generate_batch((prepared, prompts) if model_cfg.get("adapter") == "internvl" else prepared)
            gen_time = time.perf_counter() - gen_start
            if strict_inference and (len(outputs) != len(valid_samples) or any(not str(value).strip() for value in outputs)):
                raise RuntimeError(f"Strict generation coverage failed in batch {idx}: outputs={len(outputs)} expected={len(valid_samples)} empty={sum(not str(value).strip() for value in outputs)}")

            rows = []
            for sample, prompt, pred in zip(valid_samples, prompts, outputs):
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "image": sample.image,
                        "image_resolved": sample.image_resolved,
                        "prompt": prompt,
                        "prediction": pred,
                        "reference": sample.reference,
                        "dimension": sample.dimension,
                        "model": model_name,
                        "task": task_name,
                        "timestamp_utc": utc_now_iso(),
                    }
                )
            append_jsonl(predictions_path, rows)
            new_written += len(rows)
            total_written += len(rows)

            if log_batch_time:
                print(f"batch={idx} prep={prep_time:.3f}s gen={gen_time:.3f}s written={len(rows)}")

    elapsed = time.perf_counter() - started
    if strict_inference:
        final_ids = _read_completed_sample_ids(predictions_path)
        expected_ids = set(expected_sample_ids)
        if final_ids != expected_ids:
            raise RuntimeError(
                "Strict final coverage failed: "
                f"expected={len(expected_ids)} actual={len(final_ids)} "
                f"missing={len(expected_ids - final_ids)} extra={len(final_ids - expected_ids)}"
            )
    meta = {
        "model_name": model_name,
        "task": task_name,
        "sample_count": total_written,
        "new_sample_count": new_written,
        "resume": resume,
        "skipped_existing_count": initial_written,
        "elapsed_seconds": elapsed,
        "dataset_meta": dataset_meta,
        "base_config": base_cfg,
        "model_config": model_cfg,
        "text_context_protocol": {
            "protocol_name": adapter._text_context.get("protocol_name", "0shot"),
            "resolved_path": adapter._text_context.get("resolved_path", ""),
            "sha256": adapter._text_context.get("sha256", ""),
            "example_count": len(adapter._text_context.get("examples", [])),
            "selection": adapter._text_context.get("selection", {}),
        },
        "strict_inference": strict_inference,
        "runtime_snapshot": _runtime_snapshot(),
    }
    write_json(run_meta_path, meta)

    return {
        "predictions": predictions_path,
        "run_meta": run_meta_path,
    }
