import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from aesthetic_eval.data import load_eval_samples


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def row(sample_id, image="image.png", answer="synthetic reference"):
    return {
        "id": sample_id,
        "image": image,
        "conversations": [
            {"from": "human", "value": "<image> synthetic question"},
            {"from": "gpt", "value": answer},
        ],
    }


class DataTests(unittest.TestCase):
    def test_complete_synthetic_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 4), "white").save(root / "image.png")
            dataset = root / "test.jsonl"
            write_rows(dataset, [row("a")])
            samples, meta = load_eval_samples(str(dataset), str(root))
            self.assertEqual(len(samples), 1)
            self.assertEqual(meta["missing_image_count"], "0")

    def test_missing_image_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "test.jsonl"
            write_rows(dataset, [row("a", "missing.png")])
            with self.assertRaisesRegex(RuntimeError, "missing=1"):
                load_eval_samples(str(dataset), str(root))

    def test_duplicate_id_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 4), "white").save(root / "image.png")
            dataset = root / "test.jsonl"
            write_rows(dataset, [row("a"), row("a")])
            with self.assertRaisesRegex(RuntimeError, "duplicate_ids=1"):
                load_eval_samples(str(dataset), str(root))


if __name__ == "__main__":
    unittest.main()
