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

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total_rows += 1
            model_name = model_name or str(row.get("model", ""))
            score_method = score_method or str(row.get("score_method", row.get("score_source", "")))
            official_alignment = official_alignment or str(row.get("official_alignment", ""))

            gt = row.get("gt_score")
            pred = row.get("score_0_10", row.get("raw_score"))
            err = row.get("error")
            parse_status = str(row.get("parse_status", "ok"))
            if err:
                error_count += 1
                skipped_count += 1
                continue
            if parse_status != "ok":
                parse_failed_count += 1
                skipped_count += 1
                continue
            if gt is None or pred is None:
                skipped_count += 1
                continue
            try:
                gts.append(float(gt))
                preds.append(float(pred))
            except Exception:
                skipped_count += 1
                continue

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


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


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

    if len(p) == 0 or len(g) == 0:
        out = {
            "N": 0.0,
            "PLCC": 0.0,
            "SRCC": 0.0,
            "KROCC": 0.0,
            "MAE": 0.0,
            "MSE": 0.0,
            "RMSE": 0.0,
        }
        out.update(_count_metrics(total_rows, error_count, parse_failed_count, skipped_count))
        return out

    try:
        from scipy.stats import kendalltau, pearsonr, spearmanr

        plcc = float(pearsonr(p, g)[0]) if len(p) > 1 else 0.0
        srcc = float(spearmanr(p, g)[0]) if len(p) > 1 else 0.0
        krocc = float(kendalltau(p, g)[0]) if len(p) > 1 else 0.0
    except Exception:
        plcc = _safe_corr(p, g)
        srcc = _safe_corr(np.argsort(np.argsort(p)), np.argsort(np.argsort(g)))
        krocc = 0.0

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
