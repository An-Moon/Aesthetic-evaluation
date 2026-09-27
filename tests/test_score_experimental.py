import json
import sys
import tempfile
import unittest
from pathlib import Path


SCORE_SRC = Path(__file__).resolve().parents[1] / "aesthetic_eval_score_framework" / "src"
sys.path.insert(0, str(SCORE_SRC))

from aesthetic_score_eval.metrics import compute_regression_metrics, read_score_predictions


class ExperimentalScoreTests(unittest.TestCase):
    def test_score_reader_fails_on_parse_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score.jsonl"
            path.write_text(json.dumps({
                "sample_id": "synthetic-1",
                "gt_score": 5.0,
                "raw_score": None,
                "parse_status": "failed",
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "parsing failed"):
                read_score_predictions(str(path))

    def test_score_metrics_reject_constant_predictions(self):
        with self.assertRaisesRegex(ValueError, "constant"):
            compute_regression_metrics([1.0, 1.0], [1.0, 2.0])

    def test_score_metrics_reject_nonfinite_before_scipy(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compute_regression_metrics([1.0, float("nan")], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
