import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qrc_vs_qelm.models.api import FeatureBundle, QELMConfig, QRCConfig
from qrc_vs_qelm.seeds import SeedCell


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PILOT = ROOT / "scripts" / "design_pilot.py"

spec = importlib.util.spec_from_file_location("design_pilot", DESIGN_PILOT)
design_pilot = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = design_pilot
spec.loader.exec_module(design_pilot)


def _synthetic_score_rows() -> pd.DataFrame:
    rows = []
    for seed in range(8):
        rows.extend(
            [
                {
                    "gamma": 0.0,
                    "family_cell": "tau0_p1",
                    "tau": 0,
                    "p": 1,
                    "capacity_qrc": 0.40 + 0.001 * seed,
                    "capacity_qelm": 0.68 + 0.001 * seed,
                    "capacity_linear_delay": 0.48,
                },
                {
                    "gamma": 0.0,
                    "family_cell": "tau8_p3",
                    "tau": 8,
                    "p": 3,
                    "capacity_qrc": 0.74 + 0.001 * seed,
                    "capacity_qelm": 0.50 + 0.001 * seed,
                    "capacity_linear_delay": 0.42,
                },
                {
                    "gamma": 0.0,
                    "family_cell": "tau3_p2",
                    "tau": 3,
                    "p": 2,
                    "capacity_qrc": 0.60 + 0.001 * seed,
                    "capacity_qelm": 0.595 + 0.001 * seed,
                    "capacity_linear_delay": 0.50,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_aggregate_cells_uses_significance_filtered_labels_and_q_values():
    aggregate = design_pilot._aggregate_cells(
        _synthetic_score_rows(),
        seed=11,
        n_boot=128,
    )

    labels = dict(zip(aggregate["family_cell"], aggregate["label"], strict=True))
    assert labels["tau0_p1"] == "QELM"
    assert labels["tau8_p3"] == "QRC"
    assert labels["tau3_p2"] == "tie"
    assert (aggregate["ci_width"] >= 0.0).all()
    assert aggregate["q_value"].notna().all()

    sensitivity = design_pilot._sensitivity(aggregate)
    assert sensitivity["0.02"]["frac_qrc"] == 1 / 3
    assert sensitivity["0.02"]["frac_qelm"] == 1 / 3
    assert sensitivity["0.02"]["frac_tie"] == 1 / 3


def test_boundary_warning_flags_short_low_p_qrc_and_long_delay_qelm():
    aggregate = pd.DataFrame(
        [
            {"tau": 1, "p": 1, "label": "QRC"},
            {"tau": 8, "p": 3, "label": "QELM"},
            {"tau": 5, "p": 5, "label": "tie"},
        ]
    )

    warning = design_pilot._boundary_warnings(aggregate, d=5)

    assert warning["contradicts_intuition"] is True
    assert "QELM-labeled cells appear at tau >= d" in warning["warnings"]
    assert "QRC-labeled cells appear at tau < d and p <= 2" in warning["warnings"]


def test_tie_counts_split_power_limited_from_genuine_ties():
    aggregate = pd.DataFrame(
        [
            {"label": "tie", "mean_delta": 0.010, "ci_low": -0.02, "ci_high": 0.04},
            {"label": "tie", "mean_delta": 0.050, "ci_low": -0.01, "ci_high": 0.08},
            {"label": "tie", "mean_delta": -0.040, "ci_low": -0.07, "ci_high": 0.01},
            {"label": "QRC", "mean_delta": 0.050, "ci_low": 0.02, "ci_high": 0.08},
        ]
    )

    counts = design_pilot._tie_counts(aggregate, tie_band=0.02)

    assert counts == {"ties_power_limited": 2, "ties_genuine": 1}


def test_run_design_pilot_writes_json_with_injected_lightweight_builders(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        design_pilot,
        "PROFILES",
        {
            "smoke": design_pilot.PilotProfile(
                T=18,
                washout=4,
                n_boot=32,
                gamma_slices=(0.0,),
            )
        },
    )
    monkeypatch.setattr(
        design_pilot,
        "SEED_GRIDS",
        {"pilot": (SeedCell(input_seed=900, coupling_seed=987654),)},
    )
    monkeypatch.setattr(
        design_pilot,
        "grid_manifest",
        lambda name: {"name": name, "n_cells": 1, "cell_count": 1, "cells": []},
    )
    monkeypatch.setattr(
        design_pilot,
        "recommend_T_evolve",
        lambda cfg, coupling_seed: {
            "t_lo": 0.5,
            "t_recommended": 1.0,
            "t_hi": 2.0,
            "regime": "unitary",
            "timescale": 1.0,
        },
    )

    def fake_build_features(cfg, inputs, coupling_seed):
        values = np.asarray(inputs, dtype=float)
        rows = np.arange(values.shape[0], dtype=float)
        if isinstance(cfg, QRCConfig):
            features = np.column_stack([values, values**2 + 0.01 * rows])
            first_valid = 0
            model = "qrc"
        elif isinstance(cfg, QELMConfig):
            features = np.column_stack([values[::-1], values + 0.02 * rows])
            first_valid = cfg.d - 1
            model = "qelm"
        else:  # pragma: no cover - defensive test helper
            raise TypeError(type(cfg).__name__)
        return FeatureBundle(
            features=features,
            model=model,
            config=cfg.__dict__.copy(),
            couplings=np.zeros((2, 2)),
            n_obs=2,
            n_multiplex=1,
            first_valid=first_valid,
        )

    monkeypatch.setattr(design_pilot, "build_features", fake_build_features)

    result = design_pilot.run_design_pilot(
        "smoke",
        tmp_path,
        gammas=(0.2, 0.3),
        taus=(0, 1),
        degrees=(1, 2),
    )

    out_path = tmp_path / "pilot.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["profile"] == "smoke"
    assert payload["profile_config"]["gamma_slices"] == [0.2, 0.3]
    assert payload["primary_gamma"] == 0.2
    assert payload["primary_report"]["noise_power"]["n_cells"] == 4
    assert "ties_power_limited" in payload["primary_report"]
    assert "ties_genuine" in payload["primary_report"]
    assert payload["config"]["taus"] == [0, 1]
    assert payload["config"]["degrees"] == [1, 2]
    assert payload["config"]["pilot_only"] is True
    assert set(payload["slice_reports"]) == {"0.2", "0.3"}
    assert len(payload["cells"]) == 8
    assert result["seed_grid"]["name"] == "pilot"


def test_cli_parses_axis_overrides(monkeypatch, tmp_path):
    calls = {}

    def fake_run_design_pilot(profile_name, out_dir, *, gammas, taus, degrees):
        slice_reports = {
            str(float(gamma)): {
                "go": False,
                "fractions": {"frac_qrc": 0.0, "frac_qelm": 0.0, "frac_tie": 1.0},
                "ties_power_limited": 0,
                "ties_genuine": 1,
                "region_coordinates": {"QRC": [], "QELM": [], "tie": []},
            }
            for gamma in gammas
        }
        calls.update(
            {
                "profile_name": profile_name,
                "out_dir": out_dir,
                "gammas": gammas,
                "taus": taus,
                "degrees": degrees,
            }
        )
        return {
            "profile": profile_name,
            "profile_config": {
                "T": 1,
                "washout": 0,
                "n_boot": 1,
                "gamma_slices": gammas,
            },
            "status": "NO-GO",
            "primary_gamma": gammas[0],
            "passing_gammas": [],
            "primary_report": {
                "fractions": {"frac_qrc": 0.0, "frac_qelm": 0.0, "frac_tie": 1.0},
                "ties_power_limited": 0,
                "ties_genuine": 1,
                "noise_power": {
                    "median_ci_width": 0.0,
                    "max_ci_width": 0.0,
                    "n_cells_q_le_0_10": 0,
                    "n_cells": 1,
                },
                "tie_band_sensitivity": {
                    "0.02": {"frac_qrc": 0.0, "frac_qelm": 0.0, "frac_tie": 1.0}
                },
                "region_coordinates": {"QRC": [], "QELM": [], "tie": []},
                "boundary_check": {"contradicts_intuition": False, "warnings": []},
                "diagnostics": {"case": "mostly_ties"},
            },
            "slice_reports": slice_reports,
        }

    monkeypatch.setattr(design_pilot, "run_design_pilot", fake_run_design_pilot)

    out_dir = tmp_path / "pilot"
    rc = design_pilot.main(
        [
            "--profile",
            "full",
            "--gammas",
            "0.2,0.3,0.5",
            "--taus",
            "0,1,2,3,5,8,10,12,16",
            "--degrees",
            "1,2,3,5",
            "--out",
            str(out_dir),
        ]
    )

    assert rc == 0
    assert calls == {
        "profile_name": "full",
        "out_dir": out_dir,
        "gammas": (0.2, 0.3, 0.5),
        "taus": (0, 1, 2, 3, 5, 8, 10, 12, 16),
        "degrees": (1, 2, 3, 5),
    }
