#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from aesthetic_eval.adapters import build_adapter
from aesthetic_eval.config import load_yaml, merge_configs
from aesthetic_eval.data import load_eval_samples
from aesthetic_eval.io_utils import ensure_dir, write_json
from aesthetic_eval.metrics import compute_metrics, read_predictions

DEFAULT_METRICS_CONFIG = os.path.join(ROOT, "configs", "metrics", "artimuse_description_v2.yaml")


def _load_metrics_protocol(path: str) -> dict:
    protocol = load_yaml(path)
    if not bool(protocol.get("strict", False)):
        raise ValueError("Only strict metric protocols are accepted")
    models = protocol.get("models", {})
    if "clipscore" in models and "clip_cos" not in models:
        warnings.warn(
            "metrics.models.clipscore is deprecated; use metrics.models.clip_cos. "
            "The implementation is normalized CLIP image-text cosine, not CLIPScore.",
            DeprecationWarning,
            stacklevel=2,
        )
        models["clip_cos"] = models.pop("clipscore")
        protocol["legacy_alias_used"] = True
    for key in ("bertscore", "sbert", "clip_cos"):
        local_path = str(models.get(key, {}).get("local_path", ""))
        if not local_path or not os.path.isdir(local_path):
            raise FileNotFoundError(f"Missing local {key} model directory: {local_path}")
    return protocol


def _metric_versions(enabled: list) -> dict:
    names = ["sacrebleu", "rouge-score", "nltk", "bert-score", "sentence-transformers", "transformers", "torch"]
    if "spice" in {str(value).lower() for value in enabled}:
        names.append("pycocoevalcap")
    return {name: importlib.metadata.version(name) for name in names}


def _compute_from_protocol(
    pred: dict,
    enabled: list,
    protocol: dict,
    timeout: int,
    precomputed_spice_scores: list | None = None,
    metric_details: dict | None = None,
) -> dict:
    models = protocol["models"]
    return compute_metrics(
        preds=pred["preds"], refs=pred["refs"], images=pred["images"], enabled=enabled,
        bertscore_model_name=str(models["bertscore"]["local_path"]),
        bertscore_num_layers=int(models["bertscore"]["num_layers"]),
        sbert_model_name=str(models["sbert"]["local_path"]),
        clip_model_name=str(models["clip_cos"]["local_path"]),
        spice_java_path=str(protocol.get("spice", {}).get("java_path", "")),
        spice_cache_dir=str(protocol.get("spice", {}).get("cache_dir", "")),
        spice_timeout_seconds=int(protocol.get("spice", {}).get("timeout_seconds_per_chunk", 1800)),
        spice_chunk_size=int(protocol.get("spice", {}).get("chunk_size", 500)),
        spice_cache_enabled=bool(protocol.get("spice", {}).get("cache_enabled", False)),
        spice_threads=int(protocol.get("spice", {}).get("threads", 16)),
        spice_java_heap_gb=int(protocol.get("spice", {}).get("java_heap_gb", 8)),
        spice_checkpoint_enabled=bool(protocol.get("spice", {}).get("checkpoint_enabled", False)),
        spice_checkpoint_dir=str(protocol.get("spice", {}).get("checkpoint_dir", "")),
        apply_length_penalty=bool(protocol.get("length_penalty", {}).get("enabled", False)),
        clip_timeout_seconds=timeout,
        precomputed_spice_scores=precomputed_spice_scores,
        metric_details=metric_details,
    )


def _make_output_dir(base_cfg: dict, model_cfg: dict) -> str:
    out_root = str(base_cfg.get("runtime", {}).get("output_root", os.path.join(ROOT, "outputs")))
    model_name = str(model_cfg.get("model_name", "model"))
    task = str(base_cfg.get("task", "description"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"{model_name}_{task}_{ts}")
    ensure_dir(out_dir)
    return out_dir


def cmd_infer(args: argparse.Namespace) -> None:
    from aesthetic_eval.inference import configure_runtime, run_inference

    cfg = merge_configs(args.base_config, args.model_config)
    base_cfg = cfg.base
    model_cfg = cfg.model

    if args.output_root is not None:
        base_cfg.setdefault("runtime", {})["output_root"] = os.path.abspath(args.output_root)

    configure_runtime(base_cfg)

    data_cfg = base_cfg.get("data", {})
    if args.sample_limit is not None:
        data_cfg["sample_limit"] = int(args.sample_limit)
    prompt_cfg = base_cfg.get("prompt", {})

    samples, dataset_meta = load_eval_samples(
        dataset_json=str(data_cfg["dataset_json"]),
        image_root=str(data_cfg["image_root"]),
        image_alt_root=data_cfg.get("image_alt_root"),
        sample_limit=data_cfg.get("sample_limit"),
        strip_image_token=bool(prompt_cfg.get("strip_image_token", True)),
        strict_data=bool(data_cfg.get("strict_data", True)),
    )

    out_dir = os.path.abspath(args.resume_dir) if args.resume_dir else _make_output_dir(base_cfg, model_cfg)
    ensure_dir(out_dir)
    print(f"[INFO] samples={len(samples)} out_dir={out_dir}")
    if args.resume_dir:
        print(f"[INFO] resume_dir={out_dir}")

    adapter = build_adapter(base_cfg, model_cfg)
    adapter.load()

    paths = run_inference(
        adapter=adapter,
        samples=samples,
        output_dir=out_dir,
        model_name=str(model_cfg.get("model_name", "unknown")),
        task_name=str(base_cfg.get("task", "description")),
        base_cfg=base_cfg,
        model_cfg=model_cfg,
        dataset_meta=dataset_meta,
        resume=bool(args.resume_dir),
    )

    print("[DONE] inference output:")
    print(paths["predictions"])
    print(paths["run_meta"])


def cmd_eval(args: argparse.Namespace) -> None:
    protocol = _load_metrics_protocol(args.metrics_config)
    import torch
    torch.set_num_threads(int(protocol.get("cpu_threads", 8)))
    pred = read_predictions(args.pred_file)
    if args.limit is not None:
        limit = max(0, int(args.limit))
        pred = {key: values[:limit] for key, values in pred.items()}

    enabled = args.enabled or list(protocol["enabled"])
    metric_details = {}
    metrics = _compute_from_protocol(
        pred, enabled, protocol, int(args.clip_timeout), metric_details=metric_details,
    )
    metrics_by_dimension = {}
    if args.by_dimension:
        dimensions = sorted({value for value in pred.get("dimensions", []) if value})
        for dimension in dimensions:
            indices = [i for i, value in enumerate(pred["dimensions"]) if value == dimension]
            subset = {key: [values[i] for i in indices] for key, values in pred.items()}
            overall_spice = metric_details.get("spice_per_sample")
            dimension_spice = [overall_spice[i] for i in indices] if overall_spice is not None else None
            metrics_by_dimension[dimension] = _compute_from_protocol(
                subset, enabled, protocol, int(args.clip_timeout),
                precomputed_spice_scores=dimension_spice,
            )

    ensure_dir(os.path.dirname(os.path.abspath(args.output_file)) or ".")
    write_json(args.output_file, {
        "pred_file": os.path.abspath(args.pred_file),
        "limit": args.limit,
        "enabled": enabled,
        "protocol": protocol,
        "package_versions": _metric_versions(enabled),
        "metrics": metrics,
        "metrics_by_dimension": metrics_by_dimension,
    })

    print("[DONE] metrics output:")
    print(args.output_file)
    for k, v in metrics.items():
        print(f"{k}: {v}")


def _read_prediction_rows(pred_file: str) -> list:
    rows = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cmd_eval_progress(args: argparse.Namespace) -> None:
    protocol = _load_metrics_protocol(args.metrics_config)
    rows = _read_prediction_rows(args.pred_file)
    enabled = args.enabled or ["bleu", "rouge", "meteor"]
    step = max(1, int(args.step))

    checkpoints = list(range(step, len(rows) + 1, step))
    if rows and (not checkpoints or checkpoints[-1] != len(rows)):
        checkpoints.append(len(rows))

    ensure_dir(os.path.dirname(os.path.abspath(args.output_file)) or ".")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for n in checkpoints:
            chunk = rows[:n]
            pred = {
                "preds": [str(r.get("prediction", "")) for r in chunk],
                "refs": [str(r.get("reference", "")) for r in chunk],
                "images": [str(r.get("image_resolved", "")) for r in chunk],
            }
            metrics = _compute_from_protocol(pred, enabled, protocol, int(args.clip_timeout))
            item = {
                "pred_file": os.path.abspath(args.pred_file),
                "checkpoint_n": n,
                "enabled": enabled,
                "protocol_name": protocol["protocol_name"],
                "metrics": metrics,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"N={n} " + " ".join(f"{k}={v}" for k, v in metrics.items() if k != "N"))

    print("[DONE] progress metrics output:")
    print(args.output_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Aesthetic Evaluation Framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p_infer = sub.add_parser("infer", help="Run inference and save unified prediction protocol")
    p_infer.add_argument("--base-config", required=True, help="Path to shared base yaml")
    p_infer.add_argument("--model-config", required=True, help="Path to model yaml")
    p_infer.add_argument("--resume-dir", default=None, help="Existing output directory to append unfinished samples")
    p_infer.add_argument("--sample-limit", type=int, default=None, help="Limit dataset prefix; with resume, run until this total count")
    p_infer.add_argument("--output-root", default=None, help="Override runtime.output_root and record the resolved value in run metadata")
    p_infer.set_defaults(func=cmd_infer)

    p_eval = sub.add_parser("eval", help="Run offline metrics from prediction jsonl")
    p_eval.add_argument("--pred-file", required=True, help="Path to predictions.jsonl")
    p_eval.add_argument("--output-file", required=True, help="Path to metrics summary json")
    p_eval.add_argument("--metrics-config", default=DEFAULT_METRICS_CONFIG, help="Strict metric protocol yaml")
    p_eval.add_argument("--clip-timeout", type=int, default=120, help="CLIP model load timeout seconds")
    p_eval.add_argument("--limit", type=int, default=None, help="Evaluate only the first N prediction rows")
    p_eval.add_argument("--by-dimension", action="store_true", help="Also compute the same metrics separately for each non-empty dimension")
    p_eval.add_argument(
        "--enabled",
        nargs="*",
        default=None,
        help="Metric keys, e.g. bleu rouge meteor bertscore sbert_cos clip_cos",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_prog = sub.add_parser("eval-progress", help="Compute metrics on prediction prefixes")
    p_prog.add_argument("--pred-file", required=True, help="Path to predictions.jsonl")
    p_prog.add_argument("--output-file", required=True, help="Path to progress metrics jsonl")
    p_prog.add_argument("--step", type=int, default=10, help="Prefix interval, e.g. 10 gives 10/20/30...")
    p_prog.add_argument("--metrics-config", default=DEFAULT_METRICS_CONFIG, help="Strict metric protocol yaml")
    p_prog.add_argument("--clip-timeout", type=int, default=120, help="CLIP model load timeout seconds")
    p_prog.add_argument(
        "--enabled",
        nargs="*",
        default=None,
        help="Metric keys. Default uses cheap text metrics: bleu rouge meteor",
    )
    p_prog.set_defaults(func=cmd_eval_progress)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
