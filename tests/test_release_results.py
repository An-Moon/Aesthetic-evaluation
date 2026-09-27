import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v0.1.0"


def read_csv(name):
    with (RESULTS / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ReleaseResultTests(unittest.TestCase):
    def test_description_counts_and_naming(self):
        rows = read_csv("description_overall.csv")
        self.assertEqual(len(rows), 18)
        self.assertEqual({int(row["N"]) for row in rows if row["dataset"] == "ArtiMuse"}, {7984})
        self.assertEqual({int(row["N"]) for row in rows if row["dataset"] == "UNIAA"}, {500})
        self.assertTrue(all("CLIP-Cos" in row and "CLIPScore" not in row for row in rows))

    def test_dimension_coverage(self):
        rows = read_csv("artimuse_description_by_dimension.csv")
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({row["dimension"] for row in rows}), 8)
        self.assertTrue(all(int(row["N"]) == 998 for row in rows))

    def test_qa_coverage_and_hashes(self):
        rows = read_csv("uniaa_qa_overall.csv")
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(int(row["N"]) == 5354 for row in rows))
        self.assertEqual(len({row["dataset_sha256"] for row in rows}), 1)
        self.assertEqual(len({row["prediction_sha256"] for row in rows}), 9)

    def test_random_baseline_derangements(self):
        payload = json.loads((RESULTS / "uniaa_description_mismatch_5seed.json").read_text(encoding="utf-8"))
        self.assertEqual(len(payload["seeds"]), 5)
        self.assertEqual(set(payload["fixed_points_per_seed"].values()), {0})
        self.assertIn("CLIP-Cos", payload["metrics"])
        self.assertNotIn("CLIPScore", payload["metrics"])

    def test_release_manifest_hashes_match_public_files(self):
        payload = json.loads((RESULTS / "release_manifest.json").read_text(encoding="utf-8"))
        expected = {
            "artimuse_description_0shot": (
                ROOT / "configs/protocols/artimuse_description_0shot.yaml",
                ROOT / "configs/metrics/artimuse_description_v2.yaml",
            ),
            "uniaa_description": (
                ROOT / "configs/protocols/uniaa_description_1shot_greedy_v2.yaml",
                ROOT / "configs/metrics/uniaa_description_v2.yaml",
            ),
        }
        for name, (protocol, metric) in expected.items():
            with self.subTest(protocol=name):
                self.assertEqual(hashlib.sha256(protocol.read_bytes()).hexdigest(), payload["protocols"][name]["config_sha256"])
                self.assertEqual(hashlib.sha256(metric.read_bytes()).hexdigest(), payload["protocols"][name]["metric_config_sha256"])
        qa = payload["protocols"]["uniaa_perception"]
        self.assertEqual(
            hashlib.sha256((ROOT / "configs/protocols/uniaa_perception_official_v1.yaml").read_bytes()).hexdigest(),
            qa["config_sha256"],
        )
        self.assertEqual(
            hashlib.sha256((ROOT / "scripts/eval_uniaa_perception_strict.py").read_bytes()).hexdigest(),
            qa["public_parser_sha256"],
        )

    def test_model_config_hashes_match_public_files(self):
        for row in read_csv("model_matrix.csv"):
            with self.subTest(model=row["model"]):
                path = ROOT / row["config_file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["config_sha256"])


if __name__ == "__main__":
    unittest.main()
