from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrc_vs_qelm import gates, run_report


pytestmark = pytest.mark.filterwarnings(
    "ignore:An input array is constant; the correlation coefficient is not defined."
)


def _config() -> dict:
    return {
        "profile": "smoke",
        "n_qubits": 5,
        "primary": {"n_qubits": 5, "gamma": 0.3},
        "phase_diagram": {
            "taus": [0, 1, 2],
            "degrees": [1, 2],
            "gated_cells": 6,
            "uninformative_exclusion": {"both_models_test_capacity_lt": 0.05},
        },
        "seed_grid": "synthetic",
    }


def _write_manifest(stage_dir: Path, stage: int, payload: dict) -> None:
    manifest = {"core": {"payload": payload}}
    (stage_dir / f"stage{stage}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stage1_scores() -> pd.DataFrame:
    specs = [
        ("tau0_p1", 0, 1, "QRC", False),
        ("tau1_p1", 1, 1, "QRC", False),
        ("tau0_p2", 0, 2, "QELM", False),
        ("tau1_p2", 1, 2, "QELM", False),
        ("tau2_p1", 2, 1, "tie", False),
        ("tau2_p2", 2, 2, "QRC", True),
    ]
    rows: list[dict[str, object]] = []
    for seed in range(8):
        cell_id = f"cell{seed}"
        for family_cell, tau, degree, label, uninformative in specs:
            if label == "QRC" and not uninformative:
                qrc, qelm = 0.74, 0.50
            elif label == "QELM":
                qrc, qelm = 0.48, 0.73
            elif label == "tie":
                qrc, qelm = 0.60, 0.595
            else:
                qrc, qelm = 0.04, 0.01
            linear = 0.20 if degree >= 2 else 0.35
            poly = max(linear, min(0.65, qelm - 0.02))
            for model, capacity in (
                ("qrc", qrc),
                ("qelm", qelm),
                ("linear_delay", linear),
                ("poly_delay", poly),
            ):
                rows.append(
                    {
                        "cell_id": cell_id,
                        "input_seed": seed,
                        "coupling_seed": 100 + seed,
                        "shot_seed": 0,
                        "model": model,
                        "target": f"leg_{family_cell}",
                        "target_family": "phase_diagram",
                        "family_cell": family_cell,
                        "confirmatory": True,
                        "tau": tau,
                        "p": degree,
                        "capacity_test": capacity,
                        "gamma": 0.3,
                        "uninformative": uninformative,
                        "capacity_linear_delay": linear,
                        "capacity_poly_delay": poly,
                        "complete": 1,
                    }
                )
        for target, tau, degree in (
            ("stm_tau1", 1, 1),
            ("narma2", 2, 2),
            ("mackey_glass", 17, 3),
            ("parity_tau2", 2, 2),
        ):
            for model, capacity in (("qrc", 0.6), ("qelm", 0.55)):
                rows.append(
                    {
                        "cell_id": cell_id,
                        "input_seed": seed,
                        "coupling_seed": 100 + seed,
                        "shot_seed": 0,
                        "model": model,
                        "target": target,
                        "target_family": "anchor",
                        "family_cell": target,
                        "confirmatory": False,
                        "tau": tau,
                        "p": degree,
                        "capacity_test": capacity,
                        "gamma": 0.3,
                        "uninformative": False,
                        "capacity_linear_delay": 0.3,
                        "capacity_poly_delay": 0.4,
                        "complete": 1,
                    }
                )
    return pd.DataFrame(rows)


def _stage2_scores() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    shot_counts = {
        np.inf: (1, 2),
        100000.0: (2, 1),
        10000.0: (3, 1),
        1000.0: (1, 3),
    }
    family_cells = [("tau0_p1", 0, 1), ("tau0_p2", 0, 2), ("tau1_p1", 1, 1), ("tau1_p2", 1, 2)]
    for shot_seed in range(3):
        for shots, (qrc_count, qelm_count) in shot_counts.items():
            for idx, (family_cell, tau, degree) in enumerate(family_cells):
                if idx < qrc_count:
                    qrc, qelm = 0.70, 0.50
                elif idx < qrc_count + qelm_count:
                    qrc, qelm = 0.45, 0.68
                else:
                    qrc, qelm = 0.60, 0.595
                for model, capacity in (("qrc", qrc), ("qelm", qelm)):
                    rows.append(
                        {
                            "cell_id": f"cell{shot_seed}",
                            "input_seed": shot_seed,
                            "coupling_seed": 200 + shot_seed,
                            "shot_seed": shot_seed,
                            "shots": shots,
                            "model": model,
                            "target": f"leg_{family_cell}",
                            "target_family": "phase_diagram",
                            "family_cell": family_cell,
                            "confirmatory": True,
                            "tau": tau,
                            "p": degree,
                            "capacity_test": capacity,
                            "complete": 1,
                        }
                    )
    return pd.DataFrame(rows)


def _stage3_scores() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    gammas = [0.0, 0.1, 0.2, 0.3]
    dissipators = ["local_loss", "collective_loss"]
    family_cells = [("tau0_p1", 0, 1), ("tau0_p2", 0, 2), ("tau1_p1", 1, 1), ("tau1_p2", 1, 2)]
    for dissipator in dissipators:
        for gamma in gammas:
            for seed in range(2):
                for idx, (family_cell, tau, degree) in enumerate(family_cells):
                    for model in ("qrc", "qelm"):
                        qelm_catches = gamma >= 0.2 and idx < 2
                        cap_qrc = 0.62 if not qelm_catches else 0.50
                        cap_qelm = 0.50 if not qelm_catches else 0.57
                        mc = (1.0 - gamma) if model == "qrc" else (0.9 - gamma)
                        nc = gamma + (0.1 if model == "qelm" else 0.0)
                        capacity = cap_qrc if model == "qrc" else cap_qelm
                        features_dead = model == "qelm" and dissipator == "collective_loss" and gamma == 0.3
                        if features_dead:
                            mc = 0.0
                            nc = 0.0
                            capacity = 0.0
                        rows.append(
                            {
                                "cell_id": f"cell{seed}",
                                "input_seed": seed,
                                "coupling_seed": 300 + seed,
                                "shot_seed": 0,
                                "model": model,
                                "target": f"leg_{family_cell}",
                                "target_family": "phase_diagram",
                                "family_cell": family_cell,
                                "confirmatory": True,
                                "tau": tau,
                                "p": degree,
                                "capacity_test": capacity,
                                "features_dead": features_dead,
                                "gamma": gamma,
                                "dissipator": dissipator,
                                "MC": mc,
                                "NC": nc,
                                "delta_capacity_qrc_minus_qelm": np.nan,
                                "complete": 1,
                            }
                        )
    return pd.DataFrame(rows)


def _write_synthetic_stages(base: Path) -> dict[int, Path]:
    config = _config()
    out: dict[int, Path] = {}
    stage1 = base / "stage1"
    stage1.mkdir(parents=True)
    scores1 = _stage1_scores()
    scores1.to_csv(stage1 / "stage1_scores.csv", index=False)
    _write_manifest(
        stage1,
        1,
        {
            "stage": 1,
            "profile": "smoke",
            "config": config,
            "seed_grid": {"name": "synthetic", "n_cells": 8},
            "manifest_state": {
                "gates": {
                    "g1b_fraction_with_exclusion": {
                        "qrc_fraction": 0.4,
                        "qelm_fraction": 0.4,
                        "tie_fraction": 0.2,
                        "n_cells": 5,
                    },
                    "g1b_fraction_without_exclusion": {
                        "qrc_fraction": 0.5,
                        "qelm_fraction": 1.0 / 3.0,
                        "tie_fraction": 1.0 / 6.0,
                        "n_cells": 6,
                    },
                    "uninformative_cell_capacities": [
                        {
                            "family_cell": "tau2_p2",
                            "capacity_qrc_mean": 0.04,
                            "capacity_qelm_mean": 0.01,
                            "threshold": 0.05,
                        }
                    ],
                    "records": [{"gate_id": "G1b", "verdict": "INJECTED_G1B_WITH"}],
                    "records_full_unfiltered": [
                        {"gate_id": "G1b", "verdict": "INJECTED_G1B_WITHOUT"}
                    ],
                }
            },
        },
    )
    out[1] = stage1

    stage2 = base / "stage2"
    stage2.mkdir()
    _stage2_scores().to_csv(stage2 / "stage2_scores.csv", index=False)
    _write_manifest(
        stage2,
        2,
        {
            "stage": 2,
            "profile": "smoke",
            "config": config,
            "seed_grid": {"name": "synthetic", "n_cells": 3},
            "manifest_state": {},
        },
    )
    out[2] = stage2

    stage3 = base / "stage3"
    stage3.mkdir()
    _stage3_scores().to_csv(stage3 / "stage3_scores.csv", index=False)
    _write_manifest(
        stage3,
        3,
        {
            "stage": 3,
            "profile": "smoke",
            "config": config,
            "seed_grid": {"name": "synthetic", "n_cells": 2},
            "manifest_state": {},
        },
    )
    out[3] = stage3
    return out


def _run_report(stage_dirs: dict[int, Path], out: Path) -> None:
    args = ["--out", str(out)]
    for stage in sorted(stage_dirs):
        args.extend([f"--stage{stage}", str(stage_dirs[stage])])
    assert run_report.main(args) == 0


def test_run_report_all_stages_writes_figures_summary_and_report(tmp_path: Path) -> None:
    stage_dirs = _write_synthetic_stages(tmp_path / "synthetic")
    out = tmp_path / "report"

    _run_report(stage_dirs, out)

    for name in ("phase_diagram.png", "boundary_vs_shots.png", "gamma_tradeoff.png"):
        path = out / name
        assert path.exists()
        assert path.stat().st_size > 0

    summary = pd.read_csv(out / "stage_gate_summary.csv")
    assert {"stage", "gate_id", "passed", "verdict", "value", "threshold", "ci", "q"} <= set(summary.columns)
    assert set(summary["stage"]) == {1, 2, 3}

    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "INJECTED_G1B_WITH" in report
    assert "INJECTED_G1B_WITHOUT" in report
    assert "G1c with uninformative exclusion" in report
    assert "G1c without uninformative exclusion" in report
    assert "first loaded stage contains 8 executed seed cells" in report
    assert "features_dead=True" in report


def test_gate_summary_matches_stage_verdict_on_fixture(tmp_path: Path) -> None:
    stage_dirs = _write_synthetic_stages(tmp_path / "synthetic")
    out = tmp_path / "report"
    _run_report(stage_dirs, out)
    summary = pd.read_csv(out / "stage_gate_summary.csv")

    for stage in (1, 2, 3):
        loaded, missing = run_report._load_and_evaluate_stage(stage, stage_dirs[stage])
        assert missing is None
        rows = summary[summary["stage"] == stage]
        records = [
            gates.GateRecord(
                gate_id=str(row.gate_id),
                passed=bool(row.passed),
                verdict=str(row.verdict),
                value=None,
                threshold=None,
            )
            for row in rows.itertuples(index=False)
        ]
        assert gates.stage_verdict(records, stage=stage) == loaded.verdict


def test_run_report_stage1_only_degrades_gracefully(tmp_path: Path) -> None:
    stage_dirs = _write_synthetic_stages(tmp_path / "synthetic")
    out = tmp_path / "report"

    _run_report({1: stage_dirs[1]}, out)

    assert (out / "phase_diagram.png").exists()
    assert not (out / "boundary_vs_shots.png").exists()
    assert not (out / "gamma_tradeoff.png").exists()
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "Stage 2: absent" in report
    assert "Stage 3: absent" in report


def test_run_report_summary_is_deterministic(tmp_path: Path) -> None:
    stage_dirs = _write_synthetic_stages(tmp_path / "synthetic")
    left = tmp_path / "left"
    right = tmp_path / "right"

    _run_report(stage_dirs, left)
    _run_report(stage_dirs, right)

    assert (left / "stage_gate_summary.csv").read_text(encoding="utf-8") == (
        right / "stage_gate_summary.csv"
    ).read_text(encoding="utf-8")
