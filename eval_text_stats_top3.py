import json
import glob
from collections import Counter

from eval_text_stats import (
    build_aesthetic_keywords,
    build_semantic_triggers,
    build_stopwords,
    extract_text,
    match_categories,
    sentence_split,
    tokenize,
)

def evaluate_predictions(jsonl_file):
    keywords_dict = build_aesthetic_keywords()
    semantic_triggers = build_semantic_triggers()
    stopwords = build_stopwords()

    num_categories = len(keywords_dict)

    total_words = 0
    total_sentences = 0
    total_samples = 0
    total_hits = 0
    total_coverage_score = 0.0
    trigger_boosted_samples = 0

    dim_counter = Counter()
    valid_word_counter = Counter()

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                text = extract_text(item)

                words = tokenize(text)
                sentences = sentence_split(text)
                filtered_words = [w for w in words if w not in stopwords and len(w) > 1]

                total_words += len(words)
                total_sentences += len(sentences)
                total_samples += 1
                valid_word_counter.update(filtered_words)

                words_set = set(words)
                keyword_matched, matched_categories = match_categories(text, words_set, keywords_dict, semantic_triggers)

                if len(matched_categories) > len(keyword_matched):
                    trigger_boosted_samples += 1

                for category in matched_categories:
                    dim_counter[category] += 1

                hit_count = len(matched_categories)
                total_hits += hit_count
                total_coverage_score += hit_count / num_categories if num_categories else 0.0
            except Exception:
                pass

    if total_samples == 0:
        return {
            "avg_words": 0.0,
            "avg_sentences": 0.0,
            "avg_words_per_sentence": 0.0,
            "avg_dimension_hits": 0.0,
            "avg_coverage_score": 0.0,
            "semantic_boost_ratio": 0.0,
            "lexical_diversity": 0.0,
            "top3": "",
            "least3": "",
        }

    top3 = dim_counter.most_common(3)
    least3 = sorted(dim_counter.items(), key=lambda x: x[1])[:3]

    top3_formatted = ", ".join([f"{k} ({v / total_samples * 100:.1f}%)" for k, v in top3])
    least3_formatted = ", ".join([f"{k} ({v / total_samples * 100:.1f}%)" for k, v in least3])

    avg_words = total_words / total_samples
    avg_sentences = total_sentences / total_samples
    avg_words_per_sentence = total_words / total_sentences if total_sentences else 0.0
    avg_dimension_hits = total_hits / total_samples
    avg_coverage_score = total_coverage_score / total_samples
    semantic_boost_ratio = trigger_boosted_samples / total_samples

    total_valid_tokens = sum(valid_word_counter.values())
    lexical_diversity = (len(valid_word_counter) / total_valid_tokens) if total_valid_tokens else 0.0

    return {
        "avg_words": avg_words,
        "avg_sentences": avg_sentences,
        "avg_words_per_sentence": avg_words_per_sentence,
        "avg_dimension_hits": avg_dimension_hits,
        "avg_coverage_score": avg_coverage_score,
        "semantic_boost_ratio": semantic_boost_ratio,
        "lexical_diversity": lexical_diversity,
        "top3": top3_formatted,
        "least3": least3_formatted,
    }

if __name__ == "__main__":
    dirs = glob.glob("/home/Hu_xuanwei/aesthetic_eval_framework/outputs/*_description_*")
    results = {}
    for d in dirs:
        model_name = d.split('/')[-1].split('_description_')[0]
        jsonl = f"{d}/predictions.jsonl"
        metrics = evaluate_predictions(jsonl)
        results[model_name] = metrics
        print(
            f"Model: {model_name:15s} | "
            f"Words: {metrics['avg_words']:6.2f} | "
            f"Sents: {metrics['avg_sentences']:5.2f} | "
            f"Hit: {metrics['avg_dimension_hits']:4.2f} | "
            f"Cov: {metrics['avg_coverage_score']:.3f} | "
            f"Boost: {metrics['semantic_boost_ratio']*100:5.1f}% | "
            f"LexDiv: {metrics['lexical_diversity']:.3f} | "
            f"Top3: {metrics['top3']} | "
            f"Least3: {metrics['least3']}"
        )
