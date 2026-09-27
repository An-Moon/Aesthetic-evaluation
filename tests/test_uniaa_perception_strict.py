#!/usr/bin/env python3
"""Unit tests for the strict UNIAA perception answer parser."""

import unittest

from scripts.eval_uniaa_perception_strict import parse_prediction


class ParsePredictionTests(unittest.TestCase):
    def assert_parse(self, text, candidates, expected, method=None):
        parsed, actual_method, _ = parse_prediction(text, candidates)
        self.assertEqual(parsed, expected)
        if method is not None:
            self.assertEqual(actual_method, method)

    def test_bare_letter(self):
        self.assert_parse("B", ["No", "Yes"], "B", "explicit_answer")

    def test_leading_answer_with_rationale_mentions_other_option(self):
        self.assert_parse(
            "B. Yes.\nThe answer is not No because the subject is visible.",
            ["No", "Yes"],
            "B",
            "explicit_answer",
        )

    def test_final_answer_overrides_option_discussion(self):
        self.assert_parse(
            "A. Still life is unsuitable. B. Landscape is secondary. "
            "C. Portrait is less precise. Final Answer: D. Animal.",
            ["Still life", "Landscape", "Portrait", "Animal"],
            "D",
            "explicit_answer",
        )

    def test_correct_answer_markdown(self):
        self.assert_parse(
            "Reasoning mentions undersaturated and oversaturated.\n"
            "**Correct answer: B. Appropriate.**",
            ["Undersaturated", "Appropriate", "Oversaturated"],
            "B",
            "explicit_answer",
        )

    def test_best_description_on_following_line(self):
        self.assert_parse(
            "The colors are muted. Therefore, the best description is:\n\n"
            "**C. Undersaturated.**",
            ["Oversaturated", "Appropriate", "Undersaturated"],
            "C",
            "explicit_answer",
        )

    def test_most_accurate_answer(self):
        self.assert_parse(
            "The rationale discusses framing and symmetry. The most accurate answer is:\n"
            "**A. Diagonal composition.**",
            ["Diagonal composition", "Framing", "Symmetrical composition"],
            "A",
            "explicit_answer",
        )

    def test_glm_box_control_tokens(self):
        self.assert_parse(
            "The colors are balanced. Therefore, the answer is "
            "<|begin_of_box|>B<|end_of_box|>.",
            ["Oversaturated", "Appropriate", "Undersaturated"],
            "B",
            "boxed_answer",
        )

    def test_native_box_overrides_article_a(self):
        self.assert_parse(
            "The most appropriate choice is a portrait, therefore, the answer is "
            "<|begin_of_box|>B<|end_of_box|>.",
            ["Still life", "Portrait", "Animal", "Landscape"],
            "B",
            "boxed_answer",
        )

    def test_latex_box(self):
        self.assert_parse(
            "Therefore, the answer is:\n\\boxed{D}",
            ["Diagonal", "Framing", "Symmetry", "Centered"],
            "D",
            "boxed_answer",
        )

    def test_latex_text_box(self):
        self.assert_parse("\\boxed{\\text{C}}", ["No", "Maybe", "Yes"], "C", "boxed_answer")

    def test_terminal_markdown_label_and_candidate(self):
        self.assert_parse(
            "The rationale mentions oversaturated and undersaturated.\n\n**C. Appropriate.**",
            ["Oversaturated", "Undersaturated", "Appropriate"],
            "C",
            "explicit_answer",
        )

    def test_terminal_wrong_candidate_for_label_is_not_decisive(self):
        self.assert_parse("Analysis only.\nC. Appropriate.", ["Appropriate", "No", "Yes"], "A", "option_text")

    def test_best_described_by_option(self):
        self.assert_parse(
            "The composition is best described by option C: Diagonal composition.",
            ["Centered", "Framing", "Diagonal composition", "Symmetry"],
            "C",
            "explicit_answer",
        )

    def test_described_as_label(self):
        self.assert_parse(
            "The composition can be described as:\n**D. Diagonal composition.**\nThis adds movement.",
            ["Centered", "Framing", "Symmetry", "Diagonal composition"],
            "D",
            "explicit_answer",
        )

    def test_opening_label_then_rationale(self):
        self.assert_parse(
            "D. Diagonal composition. This is supported by the leading lines and subject placement.",
            ["Centered", "Framing", "Symmetry", "Diagonal composition"],
            "D",
            "explicit_answer",
        )

    def test_opening_candidate_then_option(self):
        self.assert_parse(
            "The image employs a Diagonal composition (Option C). The lines add movement.",
            ["Centered", "Framing", "Diagonal composition", "Symmetry"],
            "C",
            "explicit_answer",
        )

    def test_opening_sentence_unique_candidate(self):
        self.assert_parse(
            "The image employs a strong diagonal composition. Framing is less important here.",
            ["Centered", "Framing", "Diagonal composition", "Symmetry"],
            "C",
            "explicit_answer",
        )

    def test_opening_sentence_with_two_candidates_is_not_decisive(self):
        self.assert_parse(
            "The image is not centered composition but diagonal composition. Both require inspection.",
            ["Centered composition", "Diagonal composition", "Framing"],
            None,
            "ambiguous",
        )

    def test_opening_option_enumeration_is_not_decisive(self):
        self.assert_parse(
            "A. Centered is unlikely. B. Framing is possible. C. Diagonal composition fits.",
            ["Centered", "Framing", "Diagonal composition"],
            None,
            "ambiguous",
        )

    def test_option_enumeration_does_not_conflict_with_final_box(self):
        self.assert_parse(
            "Option A (Symmetry) does not fit. Option B (Centered) does not fit. "
            "Option C (Rule of thirds) fits. Thus, the answer is "
            "<|begin_of_box|>C<|end_of_box|>.",
            ["Symmetry", "Centered", "Rule of thirds"],
            "C",
            "boxed_answer",
        )

    def test_conflicting_explicit_answers_are_invalid(self):
        self.assert_parse(
            "Correct Answer: A. No. Final Answer: B. Yes.",
            ["No", "Yes"],
            None,
            "conflicting_explicit_answers",
        )

    def test_unique_full_option_text(self):
        self.assert_parse("Unbalanced", ["Balanced", "Unbalanced"], "B", "option_text")

    def test_nested_option_text(self):
        self.assert_parse("Light blue", ["Blue", "Light blue"], "B", "option_text")

    def test_separate_nested_mentions_are_ambiguous(self):
        self.assert_parse("Light blue, not blue", ["Blue", "Light blue"], None, "ambiguous")

    def test_multiple_options_without_final_marker_are_ambiguous(self):
        self.assert_parse("It could be No or Yes.", ["No", "Yes"], None, "ambiguous")

    def test_unknown_is_unparsed(self):
        self.assert_parse("I cannot determine this.", ["No", "Yes"], None, "unparsed")


if __name__ == "__main__":
    unittest.main()
