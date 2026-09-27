import json
from typing import Dict, List

import numpy as np


def read_score_predictions(pred_file: str) -> Dict[str, List[float]]:
    preds: List[float] = []
    gts: List[float] = []
    total_rows = 0
    error_count = 0
    parse_failed_count = 0
    skipped_count = 0
    model_name = ""
    score_method = ""
    official_alignment = ""
    sample_ids: List[str] = []

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total_rows += 1
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"Score prediction row {total_rows} has no sample_id")
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate score prediction sample_id: {sample_id}")
            sample_ids.append(sample_id)
            model_name = model_name or str(row.get("model", ""))
            score_method = score_method or str(row.get("score_method", row.get("score_source", "")))
            official_alignment = official_alignment or str(row.get("official_alignment", ""))

            gt = row.get("gt_score")
            pred = row.get("score_0_10", row.get("raw_score"))
            err = row.get("error")
            parse_status = str(row.get("parse_status", "ok"))
            if err:
                raise RuntimeError(f"Score inference failed for sample {sample_id}: {err}")
            if parse_status != "ok":
                raise RuntimeError(f"Score parsing failed for sample {sample_id}: {parse_status}")
            if gt is None or pred is None:
                raise ValueError(f"Score prediction sample {sample_id} is missing target or prediction")
            try:
                gt_value = float(gt)
                pred_value = float(pred)
            except Exception as exc:
                raise ValueError(f"Non-numeric score for sample {sample_id}") from exc
            if not np.isfinite(gt_value) or not np.isfinite(pred_value):
                raise ValueError(f"Non-finite score for sample {sample_id}")
            gts.append(gt_value)
            preds.append(pred_value)

    if total_rows == 0:
        raise ValueError("Score prediction file contains no rows")

    return {
        "preds": preds,
        "gts": gts,
        "total_rows": total_rows,
        "error_count": error_count,
        "parse_failed_count": parse_failed_count,
        "skipped_count": skipped_count,
        "model_name": model_name,
        "score_method": score_method,
        "official_alignment": official_alignment,
    }


def compute_regression_metrics(
    pred: List[float],
    gt: List[float],
    total_rows: int = 0,
    error_count: int = 0,
    parse_failed_count: int = 0,
    skipped_count: int = 0,
) -> Dict[str, float]:
    p = np.array(pred, dtype=np.float64)
    g = np.array(gt, dtype=np.float64)

    if len(p) != len(g) or len(p) < 2:
        raise ValueError(f"Regression metrics require at least two aligned values: pred={len(p)} gt={len(g)}")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(g)):
        raise ValueError("Regression metrics reject non-finite prediction or target values")
    if np.std(p) == 0 or np.std(g) == 0:
        raise ValueError("Correlation metrics are undefined for constant predictions or targets")
    from scipy.stats import kendalltau, pearsonr, spearmanr

    plcc = float(pearsonr(p, g)[0])
    srcc = float(spearmanr(p, g)[0])
    krocc = float(kendalltau(p, g)[0])

    mae = float(np.mean(np.abs(p - g)))
    mse = float(np.mean((p - g) ** 2))
    rmse = float(np.sqrt(mse))

    out = {
        "N": float(len(p)),
        "PLCC": plcc,
        "SRCC": srcc,
        "KROCC": krocc,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }
    out.update(_count_metrics(total_rows, error_count, parse_failed_count, skipped_count))
    return out


def _count_metrics(total_rows: int, error_count: int, parse_failed_count: int, skipped_count: int) -> Dict[str, float]:
    if total_rows <= 0:
        return {
            "total_rows": 0.0,
            "error_count": float(error_count),
            "parse_failed_count": float(parse_failed_count),
            "skipped_count": float(skipped_count),
            "parse_failed_rate": 0.0,
            "valid_rate": 0.0,
        }
    valid = max(0, total_rows - skipped_count)
    return {
        "total_rows": float(total_rows),
        "error_count": float(error_count),
        "parse_failed_count": float(parse_failed_count),
        "skipped_count": float(skipped_count),
        "parse_failed_rate": float(parse_failed_count / total_rows),
        "valid_rate": float(valid / total_rows),
    }
