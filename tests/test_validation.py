import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCH_CONFIGS = ROOT / "scripts" / "bench_configs.py"

spec = importlib.util.spec_from_file_location("bench_configs", BENCH_CONFIGS)
bench_configs = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(bench_configs)
project_runtime = bench_configs.project_runtime


def test_run_validation_smoke_exits_zero_and_writes_json():
    out_path = ROOT / "results_validation" / "validation.json"
    if out_path.exists():
        out_path.unlink()

    proc = subprocess.run(
        [sys.executable, "scripts/run_validation.py", "--profile", "smoke"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["profile"] == "smoke"
    assert set(payload["checks"]) == {
        "V1_qrc_physics_reproduction",
        "V2_qelm_sanity",
        "V3_matched_budget_audit",
    }
    assert all(check["passed"] is True for check in payload["checks"].values())

    v1 = payload["checks"]["V1_qrc_physics_reproduction"]
    assert v1["checks"]["stm_MC"]["source_file"].endswith("best_vs_best.json")
    assert "relaxed_smoke_directional" == v1["config"]["criterion"]

    v2 = payload["checks"]["V2_qelm_sanity"]
    assert v2["checks"]["visible_window_tau_lt_d"]["min_capacity"] >= 0.99
    assert v2["checks"]["cutoff_tau_ge_d_plus_margin"]["max_capacity"] <= 0.05
    assert v2["checks"]["features_live"]["pass"] is True


def test_project_runtime_arithmetic_on_synthetic_timing():
    timing = {
        "config_classes": {
            "qrc_n4": {"median_s": 1.5},
            "qelm_n5": {"median_s": 4.0},
        }
    }
    cells = {"qrc_n4": 3, "qelm_n5": 2}

    projected = project_runtime(timing, cells)

    assert projected["by_class"]["qrc_n4"]["projected_s"] == pytest.approx(4.5)
    assert projected["by_class"]["qelm_n5"]["projected_s"] == pytest.approx(8.0)
    assert projected["total_s"] == pytest.approx(12.5)
    assert projected["total_hours"] == pytest.approx(12.5 / 3600.0)


def test_project_runtime_accepts_flat_timing_dict():
    projected = project_runtime({"qrc_n5": 2.25}, {"qrc_n5": 4})

    assert projected["total_s"] == pytest.approx(9.0)
