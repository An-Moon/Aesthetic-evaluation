import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aesthetic_eval.metrics import _cosine_diag, _length_penalties, read_predictions


def write_predictions(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def valid_row(sample_id="1", prediction="synthetic answer"):
    return {
        "sample_id": sample_id,
        "prediction": prediction,
        "reference": "synthetic reference",
        "image_resolved": "/synthetic/image.png",
        "dimension": "synthetic",
    }


class MetricCoreTests(unittest.TestCase):
    def test_length_penalty_golden(self):
        self.assertEqual(_length_penalties(["a b", "a"], ["a b c d", "a"]), [0.5, 1.0])

    def test_cosine_diag_golden(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[0.5, 0.5], [0.0, 1.0]])
        self.assertEqual(_cosine_diag(a, b), [0.5, 1.0])

    def test_prediction_reader_rejects_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            write_predictions(path, [valid_row(prediction="")])
            with self.assertRaisesRegex(ValueError, "is empty"):
                read_predictions(str(path))

    def test_prediction_reader_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            write_predictions(path, [valid_row("1"), valid_row("1")])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_predictions(str(path))

    def test_prediction_reader_accepts_complete_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            write_predictions(path, [valid_row("1"), valid_row("2")])
            payload = read_predictions(str(path))
            self.assertEqual(len(payload["preds"]), 2)


if __name__ == "__main__":
    unittest.main()
