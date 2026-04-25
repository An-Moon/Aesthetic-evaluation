import json
import glob
import re
from collections import Counter

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return words


def sentence_split(text):
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def build_aesthetic_keywords():
    return {
        "content": [
            "narrative", "story", "storytelling", "plot", "scenario", "scene", "setting", "context",
            "environment", "character", "figure", "subject", "protagonist", "interaction", "relationship",
            "event", "moment", "situation", "depiction", "theme", "subject matter", "message", "concept",
            "visual narrative", "story arc", "narrative clarity", "story coherence", "story depth", "subject focus"
        ],
        "composition": [
            "composition", "layout", "arrangement", "structure", "organization", "balance", "symmetry", "asymmetry",
            "framing", "cropping", "positioning", "focal point", "focus", "centered", "off-center", "rule of thirds",
            "depth", "foreground", "midground", "background", "perspective", "viewpoint", "angle", "spatial",
            "visual flow", "negative space", "hierarchy", "rhythm", "diagonal", "leading lines", "visual hierarchy",
            "compositional balance", "dynamic framing", "distribution of elements", "well composed"
        ],
        "color": [
            "color", "colors", "colour", "palette", "tone", "hue", "saturation", "contrast", "brightness",
            "vibrancy", "intensity", "chromatic", "warm", "cool", "pastel", "muted", "vivid", "gradient",
            "blending", "transition", "color harmony", "color scheme", "monochrome", "complementary", "analogous",
            "triadic", "color temperature", "tonal range", "rich colors", "desaturated", "color contrast", "warm tones", "cool tones"
        ],
        "lighting": [
            "light", "lighting", "illumination", "lit", "highlight", "shadow", "shade", "contrast", "brightness",
            "darkness", "glow", "radiance", "diffuse", "soft light", "hard light", "backlighting", "rim light",
            "lighting effect", "light source", "ambient light", "dramatic lighting", "chiaroscuro", "directional light",
            "volumetric light", "high key", "low key", "shadow detail", "cinematic lighting", "natural light", "moody lighting"
        ],
        "brushstroke": [
            "line", "lines", "outline", "edge", "stroke", "brushstroke", "brushwork", "texture", "surface", "pattern",
            "detail", "fine detail", "intricate", "coarse", "smooth", "rough", "layering", "stroke direction",
            "line quality", "mark making", "hatching", "impasto", "grain", "tactile", "surface quality", "brush texture",
            "line weight", "contour", "visible strokes", "clean lines", "rough texture"
        ],
        "style": [
            "style", "artistic", "aesthetic", "visual style", "visual language", "design", "expression", "impressionistic",
            "realistic", "abstract", "minimalist", "stylized", "painterly", "rendered", "illustrative", "digital art",
            "artistic approach", "surreal", "cinematic", "contemporary", "classical", "ornamental", "stylistic",
            "signature style", "visual identity", "art direction", "illustration style", "rendering style", "visual treatment"
        ],
        "emotion": [
            "emotion", "feeling", "mood", "atmosphere", "tone", "ambience", "emotional", "expressive", "dramatic", "calm",
            "peaceful", "serene", "tense", "melancholic", "joyful", "sad", "warm", "cold", "intense", "evocative",
            "nostalgic", "mysterious", "immersive", "uplifting", "poetic", "haunting", "sentimental", "bittersweet",
            "contemplative", "tranquil", "uneasy", "dramatic tension", "emotional resonance"
        ],
        "technique": [
            "technique", "method", "execution", "anatomy", "proportion", "structure", "perspective", "depth", "rendering",
            "shading", "detailing", "refinement", "precision", "accuracy", "craftsmanship", "skillful", "well executed",
            "control", "mastery", "draftsmanship", "render quality", "technical quality", "technical control", "clean execution",
            "highly detailed", "construction", "proportional accuracy"
        ],
        "symbolism": [
            "symbol", "symbolism", "metaphor", "metaphorical", "meaning", "significance", "representation", "imply", "suggest",
            "indicate", "cultural", "icon", "motif", "allegory", "visual metaphor", "subtext", "underlying meaning",
            "archetype", "allusion", "iconography", "symbolic", "semiotic", "conceptual meaning", "deeper meaning",
            "layered meaning", "cultural reference", "symbolic layer", "encoded meaning"
        ],
        "visual_appeal": [
            "attractive", "appealing", "engaging", "eye-catching", "striking", "impressive", "beautiful", "pleasing", "memorable",
            "captivating", "compelling", "visually appealing", "aesthetic quality", "visually rich", "high quality", "polished",
            "refined", "spectacular", "gorgeous", "stunning", "visual impact", "aesthetic appeal", "high visual quality",
            "visually striking", "strong presence", "immediately engaging"
        ]
    }


def build_semantic_triggers():
    return {
        "emotion": [
            "creates a sense of", "gives a feeling of", "conveys a sense of", "evokes a feeling of", "sets the mood",
            "emotional impact", "emotionally resonant", "evokes emotion", "elicits a feeling", "emotionally charged"
        ],
        "visual_appeal": [
            "enhances the", "adds to the", "visually compelling", "draws the viewer", "catches the eye",
            "visually engaging", "strong visual impact", "more aesthetically pleasing"
        ],
        "technique": [
            "demonstrates skill", "well executed", "attention to detail", "technical proficiency",
            "shows strong control", "executed with precision", "solid technical foundation"
        ],
        "symbolism": [
            "suggests a", "symbolic meaning", "can be interpreted as", "implies that", "represents a",
            "serves as a metaphor", "deeper significance"
        ],
        "composition": [
            "contributes to", "guides the eye", "balances the scene", "creates depth",
            "strengthens the composition", "improves visual balance", "organizes the frame"
        ],
        "color": ["color palette", "use of color", "chromatic contrast", "tonal harmony"],
        "lighting": ["lighting enhances", "light and shadow", "illumination adds", "dramatic use of light"],
        "style": ["stylistic choice", "visual style", "artistic direction", "distinct style"],
        "brushstroke": ["textural quality", "brushwork adds", "line work", "surface texture"]
    }


def build_stopwords():
    return {
        "the", "and", "of", "to", "a", "in", "is", "it", "that", "this", "for", "with", "as", "on", "its", "an",
        "by", "are", "be", "from", "at", "or", "which", "their", "into", "can", "also", "has", "have", "but", "was",
        "were", "will", "would", "should", "could", "may", "might", "than", "then", "there", "these", "those", "such",
        "while", "overall", "more", "most", "very", "much", "many", "some", "any", "each", "other", "another", "both",
        "all", "his", "her", "hers", "him", "he", "she", "they", "them", "we", "our", "you", "your", "i", "me",
        "my", "mine", "us", "do", "does", "did", "done", "if", "so", "because", "therefore", "thus", "however",
        "additionally", "further", "within"
    }


def extract_text(item):
    return item.get("generated_text", item.get("text", item.get("prediction", "")))


def match_categories(text, words_set, keywords_dict, semantic_triggers):
    text_lower = text.lower()
    keyword_matched = set()

    for category, kw_list in keywords_dict.items():
        for kw in kw_list:
            if " " in kw:
                if kw in text_lower:
                    keyword_matched.add(category)
                    break
            else:
                if kw in words_set:
                    keyword_matched.add(category)
                    break

    trigger_matched = set(keyword_matched)
    for category, trigger_list in semantic_triggers.items():
        for trigger in trigger_list:
            if trigger in text_lower:
                trigger_matched.add(category)
                break

    return keyword_matched, trigger_matched

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
        }

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
            f"LexDiv: {metrics['lexical_diversity']:.3f}"
        )
