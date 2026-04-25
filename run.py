#!/usr/bin/env python3
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from aesthetic_eval.adapters import build_adapter
from aesthetic_eval.config import merge_configs
from aesthetic_eval.data import load_eval_samples
from aesthetic_eval.inference import configure_runtime, run_inference
from aesthetic_eval.io_utils import ensure_dir, write_json
from aesthetic_eval.metrics import compute_metrics, read_predictions


def _make_output_dir(base_cfg: dict, model_cfg: dict) -> str:
    out_root = str(base_cfg.get("runtime", {}).get("output_root", os.path.join(ROOT, "outputs")))
    model_name = str(model_cfg.get("model_name", "model"))
    task = str(base_cfg.get("task", "description"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"{model_name}_{task}_{ts}")
    ensure_dir(out_dir)
    return out_dir


def cmd_infer(args: argparse.Namespace) -> None:
    cfg = merge_configs(args.base_config, args.model_config)
    base_cfg = cfg.base
    model_cfg = cfg.model

    configure_runtime(base_cfg)

    data_cfg = base_cfg.get("data", {})
    prompt_cfg = base_cfg.get("prompt", {})

    samples, dataset_meta = load_eval_samples(
        dataset_json=str(data_cfg["dataset_json"]),
        image_root=str(data_cfg["image_root"]),
        image_alt_root=data_cfg.get("image_alt_root"),
        sample_limit=data_cfg.get("sample_limit"),
        strip_image_token=bool(prompt_cfg.get("strip_image_token", True)),
    )

    out_dir = _make_output_dir(base_cfg, model_cfg)
    print(f"[INFO] samples={len(samples)} out_dir={out_dir}")

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
    )

    print("[DONE] inference output:")
    print(paths["predictions"])
    print(paths["run_meta"])


def cmd_eval(args: argparse.Namespace) -> None:
    pred = read_predictions(args.pred_file)

    enabled = args.enabled
    if enabled is None:
        enabled = [
            "bleu",
            "rouge",
            "meteor",
            "bertscore",
            "sbert_cos",
            "clipscore",
        ]

    metrics = compute_metrics(
        preds=pred["preds"],
        refs=pred["refs"],
        images=pred["images"],
        enabled=enabled,
        clip_model_name=args.clip_model,
        clip_local_files_only=bool(args.clip_local_only),
        clip_timeout_seconds=int(args.clip_timeout),
    )

    ensure_dir(os.path.dirname(os.path.abspath(args.output_file)) or ".")
    write_json(args.output_file, {
        "pred_file": os.path.abspath(args.pred_file),
        "enabled": enabled,
        "metrics": metrics,
    })

    print("[DONE] metrics output:")
    print(args.output_file)
    for k, v in metrics.items():
        print(f"{k}: {v}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Aesthetic Evaluation Framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p_infer = sub.add_parser("infer", help="Run inference and save unified prediction protocol")
    p_infer.add_argument("--base-config", required=True, help="Path to shared base yaml")
    p_infer.add_argument("--model-config", required=True, help="Path to model yaml")
    p_infer.set_defaults(func=cmd_infer)

    p_eval = sub.add_parser("eval", help="Run offline metrics from prediction jsonl")
    p_eval.add_argument("--pred-file", required=True, help="Path to predictions.jsonl")
    p_eval.add_argument("--output-file", required=True, help="Path to metrics summary json")
    p_eval.add_argument("--clip-model", default="openai/clip-vit-base-patch32", help="CLIP model name")
    p_eval.add_argument("--clip-local-only", action="store_true", help="Load CLIP only from local cache/files")
    p_eval.add_argument("--clip-timeout", type=int, default=120, help="CLIP model load timeout seconds")
    p_eval.add_argument(
        "--enabled",
        nargs="*",
        default=None,
        help="Metric keys, e.g. bleu rouge meteor bertscore sbert_cos clipscore",
    )
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
