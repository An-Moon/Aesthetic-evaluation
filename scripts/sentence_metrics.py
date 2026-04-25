#!/usr/bin/env python3
"""Compute text similarity metrics for two input sentences.

Metrics:
- BLEU-1/2/3/4
- ROUGE-1/2/L (F1)
- METEOR
- BERTScore (P/R/F1)

Usage examples:
  python scripts/sentence_metrics.py --pred "the ocean on the bottom of the image is too dark." --ref "the hsjkfcahknlnlknlkwkmc lk sadjcoijox"

  python scripts/sentence_metrics.py
"""

from __future__ import annotations

import argparse
import os
import re
import signal
from typing import Dict, List


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def tokenize(text: str) -> List[str]:
    """Lightweight tokenizer for mixed Chinese/English text.

    - Chinese characters are split char-by-char.
    - English words and numbers are kept as word tokens.
    """
    text = text.strip().lower()
    if not text:
        return []
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+(?:'[a-z0-9]+)?", text)


class TimeoutError(RuntimeError):
    pass


def run_with_timeout(timeout_seconds: int, fn):
    if timeout_seconds <= 0:
        return fn()

    def handler(_signum, _frame):
        raise TimeoutError(f"operation timed out after {timeout_seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def bleu_scores(pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoother = SmoothingFunction().method1
    refs = [ref_tokens]
    return {
        "BLEU-1": float(sentence_bleu(refs, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother)),
        "BLEU-2": float(sentence_bleu(refs, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)),
        "BLEU-3": float(
            sentence_bleu(refs, pred_tokens, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoother)
        ),
        "BLEU-4": float(
            sentence_bleu(refs, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
        ),
    }


def rouge_scores(pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer

    pred_text = " ".join(pred_tokens)
    ref_text = " ".join(ref_tokens)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(ref_text, pred_text)
    return {
        "ROUGE-1": float(scores["rouge1"].fmeasure),
        "ROUGE-2": float(scores["rouge2"].fmeasure),
        "ROUGE-L": float(scores["rougeL"].fmeasure),
    }


def meteor_score_value(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    from nltk.translate.meteor_score import meteor_score

    return float(meteor_score([ref_tokens], pred_tokens))


def bertscore_values(
    pred: str,
    ref: str,
    lang: str,
    model_type: str | None = None,
    local_files_only: bool = False,
    rescale_with_baseline: bool = True,
    timeout_seconds: int = 120,
) -> Dict[str, float | str]:
    from bert_score import score as bert_score

    def _run():
        old_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
        old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        try:
            if local_files_only:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

            p, r, f1 = bert_score(
                [pred],
                [ref],
                lang=lang,
                model_type=model_type,
                rescale_with_baseline=rescale_with_baseline,
                verbose=False,
            )
        finally:
            if old_hf_hub_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_hf_hub_offline

            if old_transformers_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline

        return {
            "BERT-P": float(p[0].item()),
            "BERT-R": float(r[0].item()),
            "BERT-F1": float(f1[0].item()),
            "BERT-Rescaled": "yes" if rescale_with_baseline else "no",
        }

    try:
        return run_with_timeout(timeout_seconds, _run)
    except Exception as e:
        return {
            "BERT-P": 0.0,
            "BERT-R": 0.0,
            "BERT-F1": 0.0,
            "BERT-Error": str(e),
        }


def compute_all_metrics(
    pred: str,
    ref: str,
    bert_model_type: str | None = None,
    bert_local_files_only: bool = False,
    bert_rescale_with_baseline: bool = True,
    bert_timeout_seconds: int = 120,
) -> Dict[str, float | str]:
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)

    if not pred_tokens or not ref_tokens:
        raise ValueError("输入句子为空，或分词后为空。")

    out: Dict[str, float | str] = {}
    out.update(bleu_scores(pred_tokens, ref_tokens))
    out.update(rouge_scores(pred_tokens, ref_tokens))
    out["METEOR"] = meteor_score_value(pred_tokens, ref_tokens)

    lang = "zh" if (contains_cjk(pred) or contains_cjk(ref)) else "en"
    out.update(
        bertscore_values(
            pred,
            ref,
            lang=lang,
            model_type=bert_model_type,
            local_files_only=bert_local_files_only,
            rescale_with_baseline=bert_rescale_with_baseline,
            timeout_seconds=bert_timeout_seconds,
        )
    )
    out["BERT-Lang"] = lang
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BLEU1~4, ROUGE, METEOR and BERTScore for two sentences.")
    parser.add_argument("--pred", type=str, default=None, help="Predicted sentence")
    parser.add_argument("--ref", type=str, default=None, help="Reference sentence")
    parser.add_argument(
        "--bert-model-type",
        type=str,
        default=None,
        help="Optional model_type for BERTScore (for example: roberta-large, bert-base-chinese)",
    )
    parser.add_argument(
        "--bert-local-files-only",
        action="store_true",
        help="Use only local cached model files for BERTScore",
    )
    parser.add_argument(
        "--bert-timeout-seconds",
        type=int,
        default=120,
        help="Timeout for BERTScore calculation in seconds (0 means no timeout)",
    )
    parser.add_argument(
        "--bert-no-rescale",
        action="store_true",
        help="Disable BERTScore baseline rescaling (raw BERTScore, often numerically higher)",
    )
    args = parser.parse_args()

    pred = args.pred if args.pred is not None else input("请输入句子1（prediction）: ").strip()
    ref = args.ref if args.ref is not None else input("请输入句子2（reference）: ").strip()

    metrics = compute_all_metrics(
        pred,
        ref,
        bert_model_type=args.bert_model_type,
        bert_local_files_only=args.bert_local_files_only,
        bert_rescale_with_baseline=not args.bert_no_rescale,
        bert_timeout_seconds=args.bert_timeout_seconds,
    )

    print("\n=== Metrics ===")
    ordered_keys = [
        "BLEU-1",
        "BLEU-2",
        "BLEU-3",
        "BLEU-4",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "METEOR",
        "BERT-P",
        "BERT-R",
        "BERT-F1",
        "BERT-Rescaled",
        "BERT-Error",
        "BERT-Lang",
    ]
    for k in ordered_keys:
        if k not in metrics:
            continue
        v = metrics[k]
        if isinstance(v, str):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
