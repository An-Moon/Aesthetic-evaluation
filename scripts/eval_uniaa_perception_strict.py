#!/usr/bin/env python3
"""Strict, fail-loud scoring for UNIAA perception predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


LABELS = "ABCDE"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSON at {path}:{line_no}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_prediction(prediction: str, candidates: list[str]) -> tuple[str | None, str, list[str]]:
    raw = str(prediction).strip()
    if not raw:
        return None, "empty", []
    upper = unicodedata.normalize("NFKC", raw).upper().strip()
    valid_labels = LABELS[: len(candidates)]

    # Evidence has an explicit priority order. A boxed answer is a model-native
    # final-answer channel and therefore takes precedence over prose. This also
    # prevents English articles such as "choice is a portrait" from becoming
    # a spurious label A.
    box_hits = re.findall(
        rf"<\|BEGIN_OF_BOX\|>\s*([{valid_labels}])\s*<\|END_OF_BOX\|>", upper
    )
    box_hits.extend(
        re.findall(
            rf"\\BOXED\s*\{{\s*(?:\\TEXT\s*\{{\s*)?([{valid_labels}])"
            rf"(?:\s*\}})?\s*\}}",
            upper,
        )
    )
    box_set = sorted(set(box_hits))
    if len(box_set) == 1:
        return box_set[0], "boxed_answer", box_set
    if len(box_set) > 1:
        return None, "conflicting_boxed_answers", box_set

    # A model may discuss every option in its rationale and still provide one
    # unambiguous final answer. Only response-bound labels and labels attached
    # to an explicit answer construction are decisive. Label regexes operate
    # on the original case so the article "a" cannot be mistaken for option A.
    decisive_hits: list[str] = []
    clean_raw = raw.replace("**", "")
    first_paragraph = re.split(r"\n\s*\n", clean_raw, maxsplit=1)[0].strip()

    # Several reasoning models state the choice in the opening sentence and
    # then justify it. A response-initial "C." is decisive only when the same
    # opening paragraph does not enumerate other labelled choices.
    initial_label = re.match(rf"^\s*[\(\[]?([{valid_labels}])[\)\]]?[.\:]\s+", first_paragraph)
    paragraph_labels = re.findall(rf"(?:^|\s)[\(\[]?([{valid_labels}])[\)\]]?[.\:]\s+", first_paragraph)
    if initial_label and len(set(paragraph_labels)) == 1:
        decisive_hits.append(initial_label.group(1))

    # Also recognize an opening answer assertion in either "C. Candidate" or
    # "Candidate (Option C)" order. Candidate text must agree with the label;
    # ordinary option-by-option discussion therefore yields multiple labels
    # and remains ambiguous rather than being guessed.
    opening_candidate_hits: list[str] = []
    for index, candidate in enumerate(candidates):
        label = valid_labels[index]
        candidate_core = str(candidate).strip().rstrip(".。!！")
        candidate_pattern = re.escape(candidate_core).replace(r"\ ", r"\s+")
        if re.search(rf"\b{label}\s*[.\:]\s*{candidate_pattern}\b", first_paragraph, re.IGNORECASE):
            opening_candidate_hits.append(label)
        if re.search(
            rf"\b{candidate_pattern}\b\s*[\(\[]\s*(?:[Oo]ption\s+)?{label}\s*[\)\]]",
            first_paragraph,
            re.IGNORECASE,
        ):
            opening_candidate_hits.append(label)
    if len(set(opening_candidate_hits)) == 1:
        decisive_hits.append(opening_candidate_hits[0])

    # If the opening sentence names exactly one complete candidate, it is the
    # model's answer assertion; later mentions belong to the explanation. This
    # is stricter than choosing the first candidate anywhere in the response.
    first_sentence = re.split(r"(?<=[.!?])\s+", first_paragraph, maxsplit=1)[0]
    assertion_cue = re.search(
        r"\b(?:image|photo|picture|composition|shot|color|hue|subject)\b.*"
        r"\b(?:is|uses|employs|features|shows|depicts|characterized|described)\b",
        first_sentence,
        re.IGNORECASE,
    )
    if assertion_cue:
        sentence_tokens = normalize(first_sentence).split()
        opening_text_occurrences: list[tuple[str, int, int]] = []
        for index, candidate in enumerate(candidates):
            candidate_tokens = normalize(candidate).split()
            width = len(candidate_tokens)
            for pos in range(len(sentence_tokens) - width + 1):
                if sentence_tokens[pos : pos + width] == candidate_tokens:
                    opening_text_occurrences.append((valid_labels[index], pos, pos + width))
        opening_text_hits = {
            label
            for label, start, end in opening_text_occurrences
            if not any(
                other_start <= start and end <= other_end and other_end - other_start > end - start
                for _other_label, other_start, other_end in opening_text_occurrences
            )
        }
        if len(opening_text_hits) == 1:
            decisive_hits.append(next(iter(opening_text_hits)))
    first_line = raw.splitlines()[0].strip()
    bare_leading = re.fullmatch(
        rf"\s*(?:\*\*)?[\(\[]?([{valid_labels}])[\)\]]?[\.\:]?(?:\*\*)?\s*",
        first_line,
        flags=re.IGNORECASE,
    )
    if bare_leading:
        decisive_hits.append(bare_leading.group(1).upper())
    else:
        labelled_leading = re.fullmatch(
            rf"\s*(?:\*\*)?[\(\[]?([{valid_labels}])[\)\]]?[\.\:]?\s*(.+?)\s*(?:\*\*)?\s*",
            first_line,
            flags=re.IGNORECASE,
        )
        if labelled_leading:
            label = labelled_leading.group(1).upper()
            answer_text = labelled_leading.group(2)
            answer_text = re.sub(r"[.。!！]+$", "", answer_text).strip()
            if normalize(answer_text) == normalize(candidates[valid_labels.index(label)]):
                decisive_hits.append(label)
    marker_pattern = (
        rf"\b(?:[Ff]inal\s+[Aa]nswer|[Cc]orrect\s+[Aa]nswer|"
        rf"[Bb]est\s+(?:[Aa]nswer|[Cc]hoice|[Dd]escription)|"
        rf"[Mm]ost\s+(?:[Aa]ccurate|[Aa]ppropriate|[Ff]itting)\s+"
        rf"(?:[Aa]nswer|[Cc]hoice)|[Aa]nswer)"
        rf"\s*(?:[Ii]s\s*)?[:\-]?\s*(?:\*\*)?[\(\[]?([{valid_labels}])\b"
    )
    decisive_hits.extend(re.findall(marker_pattern, raw))
    # Unlike "Answer B", bare "Option A" commonly appears while a model
    # enumerates its rationale.  Require an explicit connector for these two
    # marker forms so option lists cannot become conflicting final answers.
    option_marker_pattern = (
        rf"\b(?:[Oo]ption|[Cc]hoice)\s*(?:[Ii]s\s*|[:\-]\s*)"
        rf"(?:\*\*)?[\(\[]?([{valid_labels}])\b"
    )
    decisive_hits.extend(re.findall(option_marker_pattern, raw))
    decisive_hits.extend(
        re.findall(
            rf"\b(?:[Bb]est\s+)?(?:[Dd]escribed|[Cc]haracterized)\s+by\s+"
            rf"(?:[Oo]ption|[Cc]hoice)\s+(?:\*\*)?([{valid_labels}])\b",
            raw,
        )
    )
    decisive_hits.extend(
        re.findall(
            rf"\b(?:[Dd]escribed|[Cc]haracterized)\s+as\s*:\s*"
            rf"(?:\*\*)?[\(\[]?([{valid_labels}])\b",
            raw,
        )
    )

    # A standalone final line like "**C. Appropriate.**" is a common output
    # form. Accept it only when its text starts with the corresponding complete
    # candidate, rather than treating any trailing letter as an answer.
    nonempty_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if nonempty_lines:
        terminal = re.fullmatch(
            rf"(?:\*\*)?\s*[\(\[]?([{valid_labels}])[\)\]]?[.\:]\s*"
            rf"(.+?)\s*(?:\*\*)?",
            nonempty_lines[-1],
        )
        if terminal:
            label = terminal.group(1)
            answer_norm = normalize(re.sub(r"[.。!！]+$", "", terminal.group(2)))
            candidate_norm = normalize(candidates[valid_labels.index(label)])
            if answer_norm == candidate_norm:
                decisive_hits.append(label)
    decisive_set = sorted(set(decisive_hits))
    if len(decisive_set) == 1:
        return decisive_set[0], "explicit_answer", decisive_set
    if len(decisive_set) > 1:
        return None, "conflicting_explicit_answers", decisive_set

    pred_tokens = normalize(raw).split()
    text_occurrences: list[tuple[str, int, int]] = []
    for index, candidate in enumerate(candidates):
        candidate_tokens = normalize(candidate).split()
        width = len(candidate_tokens)
        if width:
            for pos in range(len(pred_tokens) - width + 1):
                if pred_tokens[pos : pos + width] == candidate_tokens:
                    text_occurrences.append((valid_labels[index], pos, pos + width))
    # If one candidate is a phrase containing another candidate (e.g.
    # "Light blue" versus "Blue"), do not count the shorter occurrence that
    # lies wholly inside the longer occurrence. A separate occurrence elsewhere
    # remains ambiguous, as it should.
    retained_occurrences = []
    for occurrence in text_occurrences:
        _label, start, end = occurrence
        contained = any(
            other_start <= start and end <= other_end and (other_end - other_start) > (end - start)
            for _other_label, other_start, other_end in text_occurrences
        )
        if not contained:
            retained_occurrences.append(occurrence)
    text_set = sorted({label for label, _start, _end in retained_occurrences})

    evidence = text_set
    if len(evidence) == 1:
        return evidence[0], "option_text", evidence
    if len(evidence) > 1:
        return None, "ambiguous", evidence
    return None, "unparsed", []


def summarize(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    result = {}
    for value, group in sorted(groups.items()):
        total = len(group)
        correct = sum(bool(row["is_correct"]) for row in group)
        valid = sum(row["parsed_label"] is not None for row in group)
        result[value] = {
            "N": total,
            "correct": correct,
            "accuracy": correct / total,
            "valid": valid,
            "valid_rate": valid / total,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--min-valid-rate", type=float, default=None)
    parser.add_argument(
        "--include-details",
        action="store_true",
        help="Include per-sample predictions. Keep disabled for public release artifacts.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_file).resolve()
    pred_path = Path(args.pred_file).resolve()
    source = load_jsonl(dataset_path)
    predictions = load_jsonl(pred_path)
    source_by_id = {str(row["id"]): row for row in source}
    if len(source_by_id) != len(source):
        raise RuntimeError("Dataset contains duplicate IDs")
    pred_by_id = {str(row.get("sample_id", "")): row for row in predictions}
    if len(pred_by_id) != len(predictions) or "" in pred_by_id:
        raise RuntimeError("Predictions contain duplicate or missing sample IDs")
    expected_ids = set(source_by_id)
    actual_ids = set(pred_by_id)
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Prediction coverage mismatch: expected={len(expected_ids)} actual={len(actual_ids)} "
            f"missing={sorted(expected_ids-actual_ids)[:10]} extra={sorted(actual_ids-expected_ids)[:10]}"
        )

    details = []
    for sid in sorted(expected_ids, key=int):
        item = source_by_id[sid]
        pred = pred_by_id[sid]
        candidates = [str(value) for value in item["candidates"]]
        parsed, method, evidence = parse_prediction(str(pred.get("prediction", "")), candidates)
        correct_label = str(item["correct_label"])
        details.append(
            {
                "sample_id": sid,
                "dimension": item["dimension"],
                "question_type": item["question_type"],
                "dataset": item.get("dataset"),
                "source": item.get("source"),
                "category": item.get("category"),
                "correct_label": correct_label,
                "parsed_label": parsed,
                "parse_method": method,
                "parse_evidence": evidence,
                "is_correct": parsed == correct_label,
                "prediction": pred.get("prediction", ""),
            }
        )

    total = len(details)
    correct = sum(row["is_correct"] for row in details)
    valid = sum(row["parsed_label"] is not None for row in details)
    payload = {
        "protocol_name": "strict_uniaa_perception_v2",
        "parser_file": str(Path(__file__).resolve()),
        "parser_sha256": sha256_file(Path(__file__).resolve()),
        "pred_file": str(pred_path),
        "pred_sha256": sha256_file(pred_path),
        "dataset_file": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "N": total,
        "correct": correct,
        "accuracy": correct / total,
        "valid": valid,
        "valid_rate": valid / total,
        "invalid": total - valid,
        "parse_methods": dict(sorted(Counter(row["parse_method"] for row in details).items())),
        "prediction_label_distribution": dict(sorted(Counter(row["parsed_label"] or "INVALID" for row in details).items())),
        "by_dimension": summarize(details, "dimension"),
        "by_question_type": summarize(details, "question_type"),
        "by_dataset": summarize(details, "dataset"),
    }
    if args.include_details:
        payload["details"] = details
    output = Path(args.output_file).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"N={total} correct={correct} accuracy={correct/total:.6f} valid={valid} valid_rate={valid/total:.6f}")
    print(output)
    if args.min_valid_rate is not None and valid / total < args.min_valid_rate:
        raise RuntimeError(f"Valid-answer rate {valid/total:.4f} is below required {args.min_valid_rate:.4f}")


if __name__ == "__main__":
    main()
