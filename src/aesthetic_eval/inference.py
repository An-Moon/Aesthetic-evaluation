import os
import platform
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


def run_inference(
    adapter: BaseAdapter,
    samples: List[EvalSample],
    output_dir: str,
    model_name: str,
    task_name: str,
    base_cfg: dict,
    model_cfg: dict,
    dataset_meta: Dict[str, str],
) -> Dict[str, str]:
    batch_size = int(base_cfg.get("dataloader", {}).get("batch_size", 4))
    log_batch_time = bool(base_cfg.get("runtime", {}).get("log_batch_time", True))

    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    run_meta_path = os.path.join(output_dir, "run_meta.json")

    if os.path.exists(predictions_path):
        os.remove(predictions_path)

    batches = _chunked(samples, batch_size)
    started = time.perf_counter()
    total_written = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        if batches:
            future = executor.submit(adapter.prepare_batch, batches[0])

        for idx in tqdm(range(len(batches)), desc="infer"):
            prep_start = time.perf_counter()
            prepared, valid_samples, prompts = future.result()
            prep_time = time.perf_counter() - prep_start

            if idx + 1 < len(batches):
                future = executor.submit(adapter.prepare_batch, batches[idx + 1])

            if prepared is None or not valid_samples:
                continue

            gen_start = time.perf_counter()
            outputs = adapter.generate_batch((prepared, prompts) if model_cfg.get("adapter") == "internvl" else prepared)
            gen_time = time.perf_counter() - gen_start

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
                        "model": model_name,
                        "task": task_name,
                        "timestamp_utc": utc_now_iso(),
                    }
                )
            append_jsonl(predictions_path, rows)
            total_written += len(rows)

            if log_batch_time:
                print(f"batch={idx} prep={prep_time:.3f}s gen={gen_time:.3f}s written={len(rows)}")

    elapsed = time.perf_counter() - started
    meta = {
        "model_name": model_name,
        "task": task_name,
        "sample_count": total_written,
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
