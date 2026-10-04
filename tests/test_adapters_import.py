import importlib
import unittest


class AdapterImportTests(unittest.TestCase):
    def test_adapter_registry_import_has_no_model_load(self):
        module = importlib.import_module("aesthetic_eval.adapters")
        self.assertTrue(callable(module.build_adapter))

    def test_unknown_adapter_fails(self):
        module = importlib.import_module("aesthetic_eval.adapters")
        with self.assertRaisesRegex(ValueError, "Unknown adapter"):
            module.build_adapter({}, {"adapter": "not-a-model"})


if __name__ == "__main__":
    unittest.main()
