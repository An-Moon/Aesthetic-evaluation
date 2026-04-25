import json
import os
import signal
from typing import Dict, List

import numpy as np


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.array(values, dtype=np.float64)))


def read_predictions(pred_file: str) -> Dict[str, List[str]]:
    preds, refs, images = [], [], []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            preds.append(str(row.get("prediction", "")))
            refs.append(str(row.get("reference", "")))
            images.append(str(row.get("image_resolved", "")))
    return {"preds": preds, "refs": refs, "images": images}


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


def compute_metrics(
    preds: List[str],
    refs: List[str],
    images: List[str],
    enabled: List[str],
    clip_model_name: str,
    clip_local_files_only: bool = False,
    clip_timeout_seconds: int = 120,
) -> Dict[str, float]:
    enabled_set = {x.lower() for x in enabled}
    out: Dict[str, float] = {"N": float(len(preds))}

    if len(preds) == 0:
        return out

    if "bleu" in enabled_set:
        try:
            import sacrebleu

            out["BLEU"] = float(sacrebleu.corpus_bleu(preds, [refs]).score)
        except Exception as e:
            out["BLEU"] = 0.0
            out["BLEU-Error"] = str(e)

    if "rouge" in enabled_set:
        try:
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
        except Exception as e:
            out["ROUGE-1"] = 0.0
            out["ROUGE-2"] = 0.0
            out["ROUGE-L"] = 0.0
            out["ROUGE-Error"] = str(e)

    if "meteor" in enabled_set:
        try:
            from nltk.translate.meteor_score import meteor_score

            def _run_meteor():
                return [meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)]

            m = _run_with_timeout(600, _run_meteor)
            out["METEOR"] = _safe_mean(m)
        except Exception as e:
            out["METEOR"] = 0.0
            out["METEOR-Error"] = str(e)

    if "bertscore" in enabled_set:
        try:
            from bert_score import score as bert_score

            def _run_bertscore():
                return bert_score(preds, refs, lang="en", model_type="microsoft/deberta-base-mnli", verbose=False)

            p, r, f1 = _run_with_timeout(600, _run_bertscore)
            out["BERT-P"] = _safe_mean(p.cpu().numpy().tolist())
            out["BERT-R"] = _safe_mean(r.cpu().numpy().tolist())
            out["BERT-F1"] = _safe_mean(f1.cpu().numpy().tolist())
        except Exception as e:
            out["BERT-P"] = 0.0
            out["BERT-R"] = 0.0
            out["BERT-F1"] = 0.0
            out["BERT-Error"] = str(e)

    if "sbert_cos" in enabled_set:
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity

            def _run_sbert():
                model = SentenceTransformer("all-mpnet-base-v2")
                pe_local = model.encode(preds, convert_to_numpy=True)
                re_local = model.encode(refs, convert_to_numpy=True)
                return pe_local, re_local

            pe, re = _run_with_timeout(600, _run_sbert)
            sims = [float(cosine_similarity([a], [b])[0][0]) for a, b in zip(pe, re)]
            out["SBERT-Cos"] = _safe_mean(sims)
        except Exception as e:
            out["SBERT-Cos"] = 0.0
            out["SBERT-Error"] = str(e)

    if "clipscore" in enabled_set:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        clip_error = ""
        sims = []

        def _load_clip():
            model = CLIPModel.from_pretrained(clip_model_name, local_files_only=clip_local_files_only)
            processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=clip_local_files_only)
            return model, processor

        try:
            clip_model, clip_processor = _run_with_timeout(clip_timeout_seconds, _load_clip)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model = clip_model.to(device)
            clip_model.eval()
            max_text_len = int(getattr(clip_model.config.text_config, "max_position_embeddings", 77))
            text_chunk_len = max(1, max_text_len - 2)
            tokenizer = clip_processor.tokenizer
            sample_errors = 0

            with torch.no_grad():
                for text, image_path in zip(preds, images):
                    if not image_path or (not os.path.exists(image_path)):
                        continue
                    try:
                        image = Image.open(image_path).convert("RGB")
                    except Exception:
                        continue

                    try:
                        # CLIP text encoder has a hard max length (commonly 77 tokens).
                        # For long text, chunk by tokenizer ids and aggregate chunk embeddings.
                        token_ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
                        if not token_ids:
                            text_chunks = [""]
                        else:
                            text_chunks = []
                            for i in range(0, len(token_ids), text_chunk_len):
                                chunk_ids = token_ids[i : i + text_chunk_len]
                                text_chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))

                        image_inputs = clip_processor(images=[image], return_tensors="pt")
                        if "pixel_values" not in image_inputs:
                            continue

                        pixel_values = image_inputs["pixel_values"].to(device)
                        image_features = clip_model.get_image_features(pixel_values=pixel_values)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                        chunk_feats = []
                        for chunk_text in text_chunks:
                            text_inputs = clip_processor(
                                text=[chunk_text],
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=max_text_len,
                            )
                            if "input_ids" not in text_inputs or "attention_mask" not in text_inputs:
                                continue
                            input_ids = text_inputs["input_ids"].to(device)
                            attention_mask = text_inputs["attention_mask"].to(device)
                            text_features = clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
                            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                            chunk_feats.append(text_features)

                        if not chunk_feats:
                            continue

                        text_features = torch.cat(chunk_feats, dim=0).mean(dim=0, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        sims.append(float((text_features * image_features).sum(dim=-1).item()))
                    except Exception:
                        sample_errors += 1
                        continue

            out["CLIPScore"] = _safe_mean(sims)
            out["CLIP-N"] = float(len(sims))
            if sample_errors > 0:
                out["CLIP-Sample-Errors"] = float(sample_errors)
        except Exception as e:
            clip_error = str(e)
            out["CLIPScore"] = 0.0
            out["CLIP-N"] = 0.0
            out["CLIP-Error"] = clip_error

    return out
