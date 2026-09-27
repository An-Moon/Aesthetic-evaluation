#!/usr/bin/env python3
"""Sanitize a five-seed Sattolo mismatch summary for public release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    seeds = payload.get("seeds", [])
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Expected exactly five distinct seeds")
    fixed = payload.get("fixed_points_per_seed", {})
    if set(map(str, seeds)) != set(fixed) or any(int(value) != 0 for value in fixed.values()):
        raise ValueError("Mismatch permutations must have zero fixed points for every seed")
    metrics = dict(payload["metrics"])
    if "CLIP-Cos" not in metrics and "CLIPScore" in metrics:
        metrics["CLIP-Cos"] = metrics.pop("CLIPScore")
    public = {
        "release_version": "v0.1.0",
        "dataset": "UNIAA-Bench Description",
        "protocol": payload["protocol"],
        "N_per_seed": payload["N_per_seed"],
        "seeds": seeds,
        "fixed_points_per_seed": fixed,
        "aggregation": payload["aggregation"],
        "reported_metrics": "raw_only",
        "metrics": metrics,
        "known_limitations": [
            "The baseline measures reference-text mismatch sensitivity, not random image generation.",
            "A corresponding complete ArtiMuse five-seed CLIP-Cos run was not available at release time and is not reported.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
