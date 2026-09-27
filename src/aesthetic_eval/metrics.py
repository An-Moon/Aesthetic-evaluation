import json
import signal
import warnings
from typing import Any, Dict, List, Optional

import numpy as np


def _safe_mean(values: List[float]) -> float:
    if not values:
        raise ValueError("Cannot compute a metric mean from an empty list")
    return float(np.mean(np.array(values, dtype=np.float64)))


def _token_len(text: str) -> int:
    return len(str(text or "").split())


def _length_penalties(preds: List[str], refs: List[str]) -> List[float]:
    penalties = []
    for pred, ref in zip(preds, refs):
        pred_len, ref_len = _token_len(pred), _token_len(ref)
        penalties.append(min(pred_len, ref_len) / max(pred_len, ref_len) if max(pred_len, ref_len) else 1.0)
    return penalties


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    if len(values) != len(weights):
        raise ValueError(f"Metric/length-penalty size mismatch: {len(values)} != {len(weights)}")
    return _safe_mean([float(value) * float(weight) for value, weight in zip(values, weights)])


def read_predictions(pred_file: str) -> Dict[str, List[str]]:
    preds, refs, images, dimensions = [], [], [], []
    sample_ids = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", "")).strip()
            prediction = str(row.get("prediction", "")).strip()
            reference = str(row.get("reference", "")).strip()
            image = str(row.get("image_resolved", "")).strip()
            if not sample_id:
                raise ValueError(f"Prediction row {len(sample_ids) + 1} has no sample_id")
            if not prediction:
                raise ValueError(f"Prediction row {sample_id} is empty")
            if not reference:
                raise ValueError(f"Prediction row {sample_id} has no reference")
            if not image:
                raise ValueError(f"Prediction row {sample_id} has no resolved image path")
            sample_ids.append(sample_id)
            preds.append(prediction)
            refs.append(reference)
            images.append(image)
            dimensions.append(str(row.get("dimension", "")))
    duplicate_count = len(sample_ids) - len(set(sample_ids))
    if duplicate_count:
        raise ValueError(f"Prediction file contains {duplicate_count} duplicate sample_id value(s)")
    if not sample_ids:
        raise ValueError("Prediction file contains no samples")
    return {"preds": preds, "refs": refs, "images": images, "dimensions": dimensions}


class _TimeoutError(RuntimeError):
    pass


def _run_with_timeout(timeout_seconds: int, fn):
    if timeout_seconds is None or timeout_seconds <= 0:
        return fn()

    def _handler(_signum, _frame):
        raise _TimeoutError(f"operation timed out after {timeout_seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _encode_text_pairs_cosine_with_sentence_transformers(
    model_name: str,
    preds: List[str],
    refs: List[str],
    batch_size: int,
    device: str,
) -> List[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, local_files_only=True)
    pred_vectors = model.encode(
        preds, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    ref_vectors = model.encode(
        refs, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    return _cosine_diag(pred_vectors, ref_vectors)


def _cosine_diag(a, b) -> List[float]:
    if len(a) == 0 or len(b) == 0:
        return []
    return np.sum(a * b, axis=1).astype(np.float64).tolist()


def _feature_tensor(value):
    if hasattr(value, "pooler_output") and value.pooler_output is not None:
        return value.pooler_output
    if hasattr(value, "image_embeds") and value.image_embeds is not None:
        return value.image_embeds
    if hasattr(value, "text_embeds") and value.text_embeds is not None:
        return value.text_embeds
    return value


def compute_metrics(
    preds: List[str],
    refs: List[str],
    images: List[str],
    enabled: List[str],
    bertscore_model_name: str,
    bertscore_num_layers: int,
    sbert_model_name: str,
    clip_model_name: str,
    spice_java_path: str,
    spice_cache_dir: str,
    spice_timeout_seconds: int,
    spice_chunk_size: int,
    spice_cache_enabled: bool,
    spice_threads: int,
    spice_java_heap_gb: int,
    spice_checkpoint_enabled: bool,
    spice_checkpoint_dir: str,
    apply_length_penalty: bool,
    clip_timeout_seconds: int = 120,
    precomputed_spice_scores: Optional[List[float]] = None,
    metric_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute protocol-v1 metrics strictly and entirely from local model files.

    Any missing dependency, model, image, timeout, or metric failure raises an
    exception. A partial result must never be mistaken for a real zero score.
    """
    enabled_set = {x.lower() for x in enabled}
    if "clipscore" in enabled_set:
        warnings.warn(
            "Metric key 'clipscore' is deprecated and means raw normalized CLIP cosine. "
            "Use 'clip_cos'.",
            DeprecationWarning,
            stacklevel=2,
        )
        enabled_set.remove("clipscore")
        enabled_set.add("clip_cos")
    supported = {"bleu", "rouge", "meteor", "bertscore", "sbert_cos", "clip_cos", "spice"}
    unknown = enabled_set - supported
    if unknown:
        raise ValueError(f"Unsupported metrics: {sorted(unknown)}")
    if not (len(preds) == len(refs) == len(images)):
        raise ValueError(
            f"Metric inputs must have equal lengths: preds={len(preds)}, "
            f"refs={len(refs)}, images={len(images)}"
        )
    out: Dict[str, Any] = {"N": int(len(preds))}

    if len(preds) == 0:
        return out

    pred_lens = [_token_len(x) for x in preds]
    ref_lens = [_token_len(x) for x in refs]
    pred_len = _safe_mean([float(x) for x in pred_lens])
    ref_len = _safe_mean([float(x) for x in ref_lens])
    out["Pred-Len"] = pred_len
    out["Ref-Len"] = ref_len
    out["Len-Ratio"] = float(pred_len / ref_len) if ref_len > 0 else 0.0
    length_penalties = _length_penalties(preds, refs)
    if apply_length_penalty:
        out["Length-Penalty"] = _safe_mean(length_penalties)

    if "bleu" in enabled_set:
        from sacrebleu.metrics import BLEU

        bleu_scores = {}
        for n in range(1, 5):
            bleu_metric = BLEU(max_ngram_order=n, effective_order=True)
            bleu_scores[n] = bleu_metric.corpus_score(preds, [refs])

        # All public metrics use the [0, 1] scale in this protocol.
        out["BLEU"] = float(bleu_scores[4].score / 100.0)
        out["BLEU-1"] = float(bleu_scores[1].score / 100.0)
        out["BLEU-2"] = float(bleu_scores[2].score / 100.0)
        out["BLEU-3"] = float(bleu_scores[3].score / 100.0)
        out["BLEU-4"] = float(bleu_scores[4].score / 100.0)
        out["BLEU-BP"] = float(bleu_scores[4].bp)
        sentence_bleu = [float(BLEU(effective_order=True).sentence_score(p, [r]).score / 100.0) for p, r in zip(preds, refs)]
        if apply_length_penalty:
            out["Sentence-BLEU-Macro"] = _safe_mean(sentence_bleu)
            out["LP-BLEU"] = _weighted_mean(sentence_bleu, length_penalties)

    if "rouge" in enabled_set:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for p, r in zip(preds, refs):
            s = scorer.score(r, p)
            r1.append(float(s["rouge1"].fmeasure))
            r2.append(float(s["rouge2"].fmeasure))
            rl.append(float(s["rougeL"].fmeasure))
        out["ROUGE-1"] = _safe_mean(r1)
        out["ROUGE-2"] = _safe_mean(r2)
        out["ROUGE-L"] = _safe_mean(rl)
        if apply_length_penalty:
            out["LP-ROUGE-1"] = _weighted_mean(r1, length_penalties)
            out["LP-ROUGE-2"] = _weighted_mean(r2, length_penalties)
            out["LP-ROUGE-L"] = _weighted_mean(rl, length_penalties)

    if "meteor" in enabled_set:
        from nltk.translate.meteor_score import meteor_score

        def _run_meteor():
            return [meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)]

        m = _run_with_timeout(600, _run_meteor)
        out["METEOR"] = _safe_mean(m)
        if apply_length_penalty:
            out["LP-METEOR"] = _weighted_mean(m, length_penalties)

    if "bertscore" in enabled_set:
        import torch
        from bert_score import BERTScorer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = BERTScorer(
            model_type=bertscore_model_name, batch_size=32, device=device,
            num_layers=int(bertscore_num_layers), use_fast_tokenizer=False,
        )
        if getattr(scorer._tokenizer, "model_max_length", 512) > 1000000:
            scorer._tokenizer.model_max_length = 512
        p, r, f1 = _run_with_timeout(600, lambda: scorer.score(preds, refs))
        out["BERT-P"] = _safe_mean(p.cpu().numpy().tolist())
        out["BERT-R"] = _safe_mean(r.cpu().numpy().tolist())
        out["BERT-F1"] = _safe_mean(f1.cpu().numpy().tolist())
        if apply_length_penalty:
            out["LP-BERT-P"] = _weighted_mean(p.cpu().numpy().tolist(), length_penalties)
            out["LP-BERT-R"] = _weighted_mean(r.cpu().numpy().tolist(), length_penalties)
            out["LP-BERT-F1"] = _weighted_mean(f1.cpu().numpy().tolist(), length_penalties)

    if "sbert_cos" in enabled_set:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sims = _encode_text_pairs_cosine_with_sentence_transformers(
            sbert_model_name, preds, refs, 64, device,
        )
        out["SBERT-Cos"] = _safe_mean(sims)
        if apply_length_penalty:
            out["LP-SBERT-Cos"] = _weighted_mean(sims, length_penalties)

    if "spice" in enabled_set:
        if precomputed_spice_scores is None:
            from aesthetic_eval.spice_metric import compute_spice_strict

            spice_score, per_sample_spice = compute_spice_strict(
                preds, refs, spice_java_path, spice_cache_dir,
                timeout_seconds=spice_timeout_seconds,
                chunk_size=spice_chunk_size,
                cache_enabled=spice_cache_enabled,
                threads=spice_threads,
                java_heap_gb=spice_java_heap_gb,
                checkpoint_enabled=spice_checkpoint_enabled,
                checkpoint_dir=spice_checkpoint_dir,
            )
        else:
            if len(precomputed_spice_scores) != len(preds):
                raise ValueError(
                    f"Precomputed SPICE coverage mismatch: scores={len(precomputed_spice_scores)} preds={len(preds)}"
                )
            per_sample_spice = [float(value) for value in precomputed_spice_scores]
            spice_score = _safe_mean(per_sample_spice)
        if metric_details is not None:
            metric_details["spice_per_sample"] = list(per_sample_spice)
        out["SPICE"] = float(spice_score)
        if apply_length_penalty:
            out["LP-SPICE"] = _weighted_mean(per_sample_spice, length_penalties)

    if "clip_cos" in enabled_set:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        sims = []

        def _load_clip():
            model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True)
            processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)
            return model, processor

        clip_model, clip_processor = _run_with_timeout(clip_timeout_seconds, _load_clip)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = clip_model.to(device).eval()
        max_text_len = int(getattr(clip_model.config.text_config, "max_position_embeddings", 77))
        batch_size = 64
        pairs = [(str(text or ""), str(image_path or "")) for text, image_path in zip(preds, images)]

        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                batch_texts = [text for text, _ in batch]
                # No per-sample recovery: one bad/missing image invalidates the run.
                batch_images = [Image.open(path).convert("RGB") for _, path in batch]
                inputs = clip_processor(
                    text=batch_texts, images=batch_images, return_tensors="pt",
                    padding=True, truncation=True, max_length=max_text_len,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                image_features = _feature_tensor(clip_model.get_image_features(pixel_values=inputs["pixel_values"]))
                text_features = _feature_tensor(clip_model.get_text_features(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                ))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                sims.extend((text_features * image_features).sum(dim=-1).detach().cpu().numpy().astype(np.float64).tolist())

        if len(sims) != len(preds):
            raise RuntimeError(f"CLIP-Cos coverage mismatch: scored={len(sims)} expected={len(preds)}")
        out["CLIP-Cos"] = _safe_mean(sims)
        out["CLIP-N"] = int(len(sims))
        if apply_length_penalty:
            out["LP-CLIP-Cos"] = _weighted_mean(sims, length_penalties)

    return out
