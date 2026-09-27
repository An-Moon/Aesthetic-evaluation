import importlib


def test_adapter_registry_import_has_no_model_load():
    module = importlib.import_module("aesthetic_eval.adapters")
    assert callable(module.build_adapter)


def test_unknown_adapter_fails():
    module = importlib.import_module("aesthetic_eval.adapters")
    try:
        module.build_adapter({}, {"adapter": "not-a-model"})
    except ValueError as exc:
        assert "Unknown adapter" in str(exc)
    else:
        raise AssertionError("Unknown adapter must fail")
