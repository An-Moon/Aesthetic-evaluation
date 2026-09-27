#!/usr/bin/env python3
"""Regenerate v0.1.0 paper-ready SVG/PDF figures from aggregate CSV only."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


GENERAL = "#4C78A8"
IAA = "#C44E52"
METRICS = ["BLEU", "ROUGE-L", "METEOR", "BERT-F1", "SBERT-Cos", "SPICE", "CLIP-Cos"]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def scatter(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for ax, dataset in zip(axes, ("ArtiMuse", "UNIAA")):
        subset = [row for row in rows if row["dataset"] == dataset]
        for row in subset:
            x = sum(float(row[key]) for key in ("BLEU", "ROUGE-L", "METEOR")) / 3
            y = sum(float(row[key]) for key in ("BERT-F1", "SBERT-Cos")) / 2
            color = GENERAL if row["group"] == "General MLLMs" else IAA
            ax.scatter(x, y, s=130, color=color, alpha=0.82, edgecolor="white", linewidth=0.8)
            ax.annotate(row["model"], (x, y), xytext=(4, 5), textcoords="offset points", fontsize=7.2)
        ax.set_title(f"{dataset}: Lexical vs. Semantic Alignment", fontweight="bold")
        ax.set_xlabel("Lexical alignment (mean BLEU, R-L, METEOR)")
        ax.set_ylabel("Semantic alignment (mean BERT-F1, SBERT-Cos)")
    handles = [
        mpl.patches.Patch(color=GENERAL, label="General MLLMs"),
        mpl.patches.Patch(color=IAA, label="IAA-specific MLLMs"),
    ]
    axes[0].legend(handles=handles, loc="best", frameon=True)
    fig.tight_layout()
    save(fig, out, "description_lexical_semantic_scatter")


def group_delta(rows: list[dict], out: Path) -> None:
    means = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in METRICS:
            means[(row["dataset"], row["group"])][metric].append(float(row[metric]))
    deltas = {}
    for dataset in ("ArtiMuse", "UNIAA"):
        deltas[dataset] = []
        for metric in METRICS:
            general = sum(means[(dataset, "General MLLMs")][metric]) / len(means[(dataset, "General MLLMs")][metric])
            iaa = sum(means[(dataset, "IAA-specific MLLMs")][metric]) / len(means[(dataset, "IAA-specific MLLMs")][metric])
            deltas[dataset].append(iaa - general)
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    y = range(len(METRICS))
    for dataset, offset, color, marker in (("ArtiMuse", -0.12, IAA, "o"), ("UNIAA", 0.12, GENERAL, "s")):
        yy = [value + offset for value in y]
        ax.scatter(deltas[dataset], yy, s=52, color=color, marker=marker, label=dataset, zorder=3)
        for x, yvalue in zip(deltas[dataset], yy):
            ax.plot([0, x], [yvalue, yvalue], color=color, alpha=0.45)
            ax.annotate(f"{x:+.3f}", (x, yvalue), xytext=(4 if x >= 0 else -4, 0),
                        textcoords="offset points", ha="left" if x >= 0 else "right", va="center", fontsize=7)
    ax.axvline(0, color="#374151", linewidth=1)
    all_delta = deltas["ArtiMuse"] + deltas["UNIAA"]
    ax.set_xlim(min(all_delta) - 0.008, max(all_delta) + 0.008)
    ax.set_yticks(list(y), ["R-L" if value == "ROUGE-L" else value for value in METRICS])
    ax.invert_yaxis()
    ax.set_xlabel(r"Group difference ($\Delta=$ IAA-specific $-$ General)")
    ax.set_title("Specialization Gain Reverses Across Benchmarks", fontweight="bold")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.34), frameon=False)
    fig.tight_layout()
    save(fig, out, "description_cross_dataset_group_delta")


def qa_bar(rows: list[dict], out: Path) -> None:
    rows = sorted(rows, key=lambda row: float(row["accuracy"]))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    colors = [GENERAL if row["group"] == "General MLLMs" else IAA for row in rows]
    values = [100 * float(row["accuracy"]) for row in rows]
    bars = ax.barh([row["model"] for row in rows], values, color=colors, alpha=0.82)
    for bar, value in zip(bars, values):
        ax.text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8)
    ax.set_xlim(min(values) - 2.5, max(values) + 3.0)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("UNIAA-Bench Perception (strict-v2)", fontweight="bold")
    ax.legend(handles=[mpl.patches.Patch(color=GENERAL, label="General MLLMs"),
                       mpl.patches.Patch(color=IAA, label="IAA-specific MLLMs")], loc="lower right")
    fig.tight_layout()
    save(fig, out, "uniaa_qa_accuracy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/v0.1.0"))
    args = parser.parse_args()
    style()
    figures = args.results_dir / "figures"
    description = read_csv(args.results_dir / "description_overall.csv")
    qa = read_csv(args.results_dir / "uniaa_qa_overall.csv")
    scatter(description, figures)
    group_delta(description, figures)
    qa_bar(qa, figures)


if __name__ == "__main__":
    main()
