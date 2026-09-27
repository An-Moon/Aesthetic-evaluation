import re
from typing import Optional


def extract_numeric_score(text: str, min_score: float = 1.0, max_score: float = 10.0) -> Optional[float]:
    lines = str(text).strip().split("\n")
    for line in reversed(lines):
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if matches:
            score = float(matches[-1])
            return float(max(min_score, min(max_score, score)))

    parts = re.split(r"assistant", str(text), flags=re.IGNORECASE)
    if len(parts) > 1:
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", parts[-1])
        if matches:
            score = float(matches[-1])
            return float(max(min_score, min(max_score, score)))

    return None
