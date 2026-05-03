import glob
import json
import os
from typing import Dict, List


def build_leaderboard(metrics_glob: str) -> Dict:
    files = sorted(glob.glob(metrics_glob))
    rows: List[Dict] = []

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        metrics = payload.get("metrics", {})
        rows.append(
            {
                "metrics_file": os.path.abspath(path),
                "model": payload.get("model", ""),
                "score_method": payload.get("score_method", ""),
                "official_alignment": payload.get("official_alignment", ""),
                "N": metrics.get("N", 0.0),
                "total_rows": metrics.get("total_rows", metrics.get("N", 0.0)),
                "PLCC": metrics.get("PLCC", 0.0),
                "SRCC": metrics.get("SRCC", 0.0),
                "KROCC": metrics.get("KROCC", 0.0),
                "MAE": metrics.get("MAE", 0.0),
                "RMSE": metrics.get("RMSE", 0.0),
                "error_count": metrics.get("error_count", 0.0),
                "parse_failed_count": metrics.get("parse_failed_count", 0.0),
                "parse_failed_rate": metrics.get("parse_failed_rate", 0.0),
            }
        )

    rows = sorted(rows, key=lambda x: (x.get("SRCC", 0.0), x.get("PLCC", 0.0)), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    return {
        "count": len(rows),
        "items": rows,
    }
