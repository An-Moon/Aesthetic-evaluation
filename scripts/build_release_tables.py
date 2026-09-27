#!/usr/bin/env python3
"""Build compact LaTeX tables from aggregate release CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def esc(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_")


def description(rows: list[dict], dataset: str) -> str:
    part = [row for row in rows if row["dataset"] == dataset]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\caption{" + esc(dataset) + r" reference-based Description results. Raw metrics only.}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Model & Len. & BLEU & R-L & METEOR & BERT-F1 & SBERT & SPICE & CLIP-Cos \\",
        r"\midrule",
    ]
    previous = None
    for row in part:
        if previous is not None and row["group"] != previous:
            lines.append(r"\midrule")
        previous = row["group"]
        vals = [float(row[key]) for key in ("Pred-Len", "BLEU", "ROUGE-L", "METEOR", "BERT-F1", "SBERT-Cos", "SPICE", "CLIP-Cos")]
        lines.append(f"{esc(row['model'])} & {vals[0]:.2f} & " + " & ".join(f"{v:.4f}" for v in vals[1:]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def qa(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda row: float(row["accuracy"]), reverse=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{UNIAA-Bench Perception results with strict-v2 parsing.}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Accuracy (\%) & Valid (\%) & Correct & Invalid \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{esc(row['model'])} & {100*float(row['accuracy']):.2f} & "
            f"{100*float(row['valid_rate']):.2f} & {int(row['correct'])} & {int(row['invalid'])} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/v0.1.0"))
    args = parser.parse_args()
    output = args.results_dir / "tables"
    output.mkdir(parents=True, exist_ok=True)
    desc = read(args.results_dir / "description_overall.csv")
    (output / "artimuse_description.tex").write_text(description(desc, "ArtiMuse"), encoding="utf-8")
    (output / "uniaa_description.tex").write_text(description(desc, "UNIAA"), encoding="utf-8")
    (output / "uniaa_qa.tex").write_text(qa(read(args.results_dir / "uniaa_qa_overall.csv")), encoding="utf-8")


if __name__ == "__main__":
    main()
