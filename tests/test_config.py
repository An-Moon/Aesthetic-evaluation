import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aesthetic_eval.config import load_yaml, parse_device_map


class ConfigTests(unittest.TestCase):
    def test_recursive_env_and_home_expansion(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"AESTHETIC_TEST_ROOT": directory}):
            path = Path(directory) / "config.yaml"
            path.write_text("root: ${AESTHETIC_TEST_ROOT}\nnested:\n  - ~/model\n", encoding="utf-8")
            value = load_yaml(str(path))
            self.assertEqual(value["root"], directory)
            self.assertEqual(value["nested"][0], os.path.expanduser("~/model"))

    def test_missing_env_fails(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "config.yaml"
            path.write_text("root: ${AESTHETIC_REQUIRED_TEST}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unresolved required"):
                load_yaml(str(path))

    def test_shell_default_expression_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"AESTHETIC_TEST_ROOT": directory}):
            path = Path(directory) / "config.yaml"
            path.write_text("root: ${AESTHETIC_TEST_ROOT:-fallback}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unresolved environment expression"):
                load_yaml(str(path))

    def test_device_map_json(self):
        self.assertEqual(parse_device_map('{"": 0}'), {"": 0})
        self.assertEqual(parse_device_map("auto"), "auto")


if __name__ == "__main__":
    unittest.main()
