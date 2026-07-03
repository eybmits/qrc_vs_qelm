import importlib.util
from pathlib import Path


def _load_smoke_summary():
    smoke_path = Path(__file__).resolve().parent.parent / "scripts" / "smoke_core.py"
    spec = importlib.util.spec_from_file_location("smoke_core", smoke_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.smoke_summary


def test_smoke_core_deterministic() -> None:
    smoke_summary = _load_smoke_summary()
    first = smoke_summary()
    second = smoke_summary()

    assert first == second
    assert first[0] == (150, 30)
    assert len(first[1]) == 16
