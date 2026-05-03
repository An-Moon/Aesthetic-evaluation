#!/usr/bin/env python3
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from aesthetic_score_eval.adapters import build_adapter
from aesthetic_score_eval.config import merge_configs
from aesthetic_score_eval.data import load_score_samples
from aesthetic_score_eval.inference import configure_runtime, run_score_inference
from aesthetic_score_eval.io_utils import ensure_dir, write_json
from aesthetic_score_eval.metrics import compute_regression_metrics, read_score_predictions
from aesthetic_score_eval.report import build_leaderboard


def _make_output_dir(base_cfg: dict, model_cfg: dict) -> str:
    out_root = str(base_cfg.get("runtime", {}).get("output_root", os.path.join(ROOT, "outputs")))
    model_name = str(model_cfg.get("model_name", "model"))
    task = str(base_cfg.get("task", "score"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"{model_name}_{task}_{ts}")
    ensure_dir(out_dir)
    return out_dir


def cmd_infer_score(args: argparse.Namespace) -> None:
    cfg = merge_configs(args.base_config, args.model_config)
    base_cfg = cfg.base
    model_cfg = cfg.model

    configure_runtime(base_cfg)

    data_cfg = base_cfg.get("data", {})
    samples, dataset_meta = load_score_samples(
        dataset_jsonl=str(data_cfg["dataset_jsonl"]),
        image_root=str(data_cfg.get("image_root", "")),
        image_alt_root=data_cfg.get("image_alt_root"),
        sample_limit=data_cfg.get("sample_limit"),
        sample_size=data_cfg.get("sample_size"),
        seed=int(base_cfg.get("seed", 42)),
        shuffle=bool(data_cfg.get("shuffle", True)),
        sample_manifest_out=data_cfg.get("sample_manifest_out"),
        sample_manifest_in=data_cfg.get("sample_manifest_in"),
    )

    out_dir = _make_output_dir(base_cfg, model_cfg)
    print(f"[INFO] samples={len(samples)} out_dir={out_dir}")

    adapter = build_adapter(base_cfg, model_cfg)
    adapter.load()

    paths = run_score_inference(
        adapter=adapter,
        samples=samples,
        output_dir=out_dir,
        base_cfg=base_cfg,
        model_cfg=model_cfg,
        dataset_meta=dataset_meta,
    )

    print("[DONE] inference output:")
    print(paths["predictions"])
    print(paths["run_meta"])


def cmd_eval_score(args: argparse.Namespace) -> None:
    payload = read_score_predictions(args.pred_file)
    metrics = compute_regression_metrics(
        payload["preds"],
        payload["gts"],
        total_rows=int(payload.get("total_rows", 0)),
        error_count=int(payload.get("error_count", 0)),
        parse_failed_count=int(payload.get("parse_failed_count", 0)),
        skipped_count=int(payload.get("skipped_count", 0)),
    )

    ensure_dir(os.path.dirname(os.path.abspath(args.output_file)) or ".")
    write_json(
        args.output_file,
        {
            "pred_file": os.path.abspath(args.pred_file),
            "model": payload.get("model_name", ""),
            "score_method": payload.get("score_method", ""),
            "official_alignment": payload.get("official_alignment", ""),
            "metrics": metrics,
        },
    )

    print("[DONE] metrics output:")
    print(args.output_file)
    for k, v in metrics.items():
        print(f"{k}: {v}")


def cmd_report(args: argparse.Namespace) -> None:
    board = build_leaderboard(args.metrics_glob)
    ensure_dir(os.path.dirname(os.path.abspath(args.output_file)) or ".")
    write_json(args.output_file, board)
    print("[DONE] report output:")
    print(args.output_file)


def cmd_validate_score(args: argparse.Namespace) -> None:
    cfg = merge_configs(args.base_config, args.model_config)
    base_cfg = cfg.base
    model_cfg = cfg.model

    configure_runtime(base_cfg)

    data_cfg = base_cfg.get("data", {})
    samples, dataset_meta = load_score_samples(
        dataset_jsonl=str(data_cfg["dataset_jsonl"]),
        image_root=str(data_cfg.get("image_root", "")),
        image_alt_root=data_cfg.get("image_alt_root"),
        sample_limit=data_cfg.get("sample_limit"),
        sample_size=args.sample_size if args.sample_size is not None else data_cfg.get("sample_size"),
        seed=int(base_cfg.get("seed", 42)),
        shuffle=bool(data_cfg.get("shuffle", True)),
        sample_manifest_out=data_cfg.get("sample_manifest_out"),
        sample_manifest_in=data_cfg.get("sample_manifest_in"),
    )

    out_dir = _make_output_dir(base_cfg, model_cfg)
    print(f"[INFO] validate samples={len(samples)} out_dir={out_dir}")

    adapter = build_adapter(base_cfg, model_cfg)
    adapter.load()
    paths = run_score_inference(
        adapter=adapter,
        samples=samples,
        output_dir=out_dir,
        base_cfg=base_cfg,
        model_cfg=model_cfg,
        dataset_meta=dataset_meta,
    )

    payload = read_score_predictions(paths["predictions"])
    metrics = compute_regression_metrics(
        payload["preds"],
        payload["gts"],
        total_rows=int(payload.get("total_rows", 0)),
        error_count=int(payload.get("error_count", 0)),
        parse_failed_count=int(payload.get("parse_failed_count", 0)),
        skipped_count=int(payload.get("skipped_count", 0)),
    )
    metrics_path = os.path.join(out_dir, "metrics.json")
    write_json(
        metrics_path,
        {
            "pred_file": os.path.abspath(paths["predictions"]),
            "model": payload.get("model_name", ""),
            "score_method": payload.get("score_method", ""),
            "official_alignment": payload.get("official_alignment", ""),
            "metrics": metrics,
        },
    )
    print("[DONE] validate output:")
    print(paths["predictions"])
    print(metrics_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLLM Aesthetic Score Evaluation Toolbox")
    sub = parser.add_subparsers(dest="command", required=True)

    p_infer = sub.add_parser("infer-score", help="Run score inference and save unified score protocol")
    p_infer.add_argument("--base-config", required=True, help="Path to shared base yaml")
    p_infer.add_argument("--model-config", required=True, help="Path to model yaml")
    p_infer.set_defaults(func=cmd_infer_score)

    p_eval = sub.add_parser("eval-score", help="Run regression metrics from score predictions jsonl")
    p_eval.add_argument("--pred-file", required=True, help="Path to predictions.jsonl")
    p_eval.add_argument("--output-file", required=True, help="Path to metrics summary json")
    p_eval.set_defaults(func=cmd_eval_score)

    p_report = sub.add_parser("report", help="Aggregate metric json files into one leaderboard")
    p_report.add_argument("--metrics-glob", required=True, help="Glob for metric summary files")
    p_report.add_argument("--output-file", required=True, help="Output leaderboard json path")
    p_report.set_defaults(func=cmd_report)

    p_validate = sub.add_parser("validate-score", help="Run score inference and metrics in one command")
    p_validate.add_argument("--base-config", required=True, help="Path to shared base yaml")
    p_validate.add_argument("--model-config", required=True, help="Path to model yaml")
    p_validate.add_argument("--sample-size", type=int, default=None, help="Optional override for validation sample size")
    p_validate.set_defaults(func=cmd_validate_score)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
