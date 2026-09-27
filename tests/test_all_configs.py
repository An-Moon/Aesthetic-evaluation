import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from aesthetic_eval.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]
SCORE_SRC = ROOT / "aesthetic_eval_score_framework" / "src"
sys.path.insert(0, str(SCORE_SRC))
from aesthetic_score_eval.config import load_yaml as load_score_yaml


ENV = {
    "AESTHETIC_DATA_ROOT": "/tmp/aesthetic-data",
    "AESTHETIC_MODEL_ROOT": "/tmp/aesthetic-models",
    "AESTHETIC_OUTPUT_ROOT": "/tmp/aesthetic-outputs",
    "AESTHETIC_METRIC_MODEL_ROOT": "/tmp/aesthetic-metrics",
    "AESTHETIC_JAVA": "/tmp/java",
    "ARTIMUSE_CONTEXT_FILE": "/tmp/artimuse-context.json",
    "UNIAA_1SHOT_CONTEXT_FILE": "/tmp/uniaa-context.json",
}


class AllConfigTests(unittest.TestCase):
    def test_root_configs_resolve(self):
        with patch.dict(os.environ, ENV, clear=False):
            paths = sorted((ROOT / "configs").rglob("*.yaml"))
            self.assertGreater(len(paths), 0)
            for path in paths:
                with self.subTest(path=path):
                    self.assertIsInstance(load_yaml(str(path)), dict)

    def test_experimental_score_configs_resolve(self):
        with patch.dict(os.environ, ENV, clear=False):
            paths = sorted((ROOT / "aesthetic_eval_score_framework" / "configs").rglob("*.yaml"))
            self.assertGreater(len(paths), 0)
            for path in paths:
                with self.subTest(path=path):
                    self.assertIsInstance(load_score_yaml(str(path)), dict)


if __name__ == "__main__":
    unittest.main()
