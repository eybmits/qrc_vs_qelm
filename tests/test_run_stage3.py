from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrc_vs_qelm import run_stage1, run_stage3
from qrc_vs_qelm.manifests import sha256_frame
from qrc_vs_qelm.seeds import SeedCell


def _tiny_config() -> dict:
    return {
        "profile": "smoke",
        "n_qubits": 2,
        "T": 48,
        "washout": 5,
        "primary": {
            "n_qubits": 2,
            "gamma": 0.2,
            "dt": 0.5,
            "h": 0.4,
            "J_s": 0.7,
            "dissipator": "local_loss",
            "n_virtual": 1,
            "n_snapshots": 1,
            "multiplexing": 1,
            "quantize_levels": 5,
            "observables": {
                "family": "pauli",
                "axes": ["x", "y", "z"],
                "max_weight": 1,
                "dimension": 6,
            },
        },
        "phase_diagram": {
            "family": "legendre_delay",
            "taus": [0, 1],
            "degrees": [1],
            "gated_cells": 2,
            "robustness_degrees": [1, 2],
            "robustness_cells": 4,
            "uninformative_exclusion": {"both_models_test_capacity_lt": 0.05},
        },
        "secondary_product_family": {
            "enabled": False,
            "taus": [],
            "degree_pairs": [],
        },
        "anchors": {"gated": False, "families": []},
        "hp_grids": {
            "alpha": [0.0, 1.0e-4],
            "qrc": {"dt": [0.5], "h_in": [1.0], "cardinality": 1},
            "qelm": {
                "d": [1],
                "T_evolve": {
                    "mode": "recommend_T_evolve_window_multipliers",
                    "multipliers_of_timescale": [1.0],
                    "constraint": "t_lo <= T_evolve <= t_hi",
                },
                "cardinality": 1,
            },
        },
        "seed_grid": "stage1_smoke",
        "seed_grids": {
            "stage1": "stage1_smoke",
            "stage2": "stage2_shots",
            "stage3": "stage3_gamma",
        },
        "gamma_grid": [0.05, 0.2],
        "dissipator_grid": ["local_loss"],
        "runtime_budget": {
            "stage_budget_hours": 10.0,
            "active_shrink": {"stage3": {"applied": False}},
        },
    }


@pytest.fixture
def tiny_stage3(monkeypatch: pytest.MonkeyPatch) -> dict:
    config = _tiny_config()
    monkeypatch.setattr(run_stage3.run_stage1, "_load_config", lambda profile: copy.deepcopy(config))
    monkeypatch.setattr(
        run_stage3.manifests,
        "require_clean_tree",
        lambda allow_dirty: {"preregistration_violated": False},
    )
    cells = (SeedCell(input_seed=0, coupling_seed=314159),)
    monkeypatch.setattr(run_stage3, "SEED_GRIDS", {"stage3_gamma": cells})
    monkeypatch.setattr(
        run_stage3,
        "grid_manifest",
        lambda name: {
            "name": name,
            "n_cells": len(cells),
            "cell_count": len(cells),
            "cells": [cell.__dict__ for cell in cells],
        },
    )
    return config


def _run(out: Path) -> pd.DataFrame:
    rc = run_stage3.main(
        [
            "--profile",
            "smoke",
            "--out",
            str(out),
            "--checkpoint-every",
            "1",
        ]
    )
    assert rc == 0
    return pd.read_csv(out / "stage3_scores.csv", float_precision="round_trip")


def test_stage3_smoke_main_writes_scores_and_manifest(tiny_stage3, tmp_path: Path) -> None:
    out = tmp_path / "stage3"
    scores = _run(out)
    manifest = json.loads((out / "stage3_manifest.json").read_text(encoding="utf-8"))
    payload = manifest["core"]["payload"]

    assert not scores.empty
    assert {"gamma", "dissipator", "MC", "NC"} <= set(scores.columns)
    assert {"qrc", "qelm"} == set(scores["model"])
    assert payload["stage"] == 3
    assert payload["profile"] == "smoke"
    assert payload["outputs"]["n_score_rows"] == len(scores)
    assert payload["manifest_state"]["feature_cache"]
    assert (out / "checkpoints").is_dir()


def test_stage3_smoke_determinism(tiny_stage3, tmp_path: Path) -> None:
    left = _run(tmp_path / "left")
    right = _run(tmp_path / "right")

    assert sha256_frame(left) == sha256_frame(right)


def test_stage3_clean_tree_gate_blocks_dirty_without_allow(
    tiny_stage3,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_require_clean_tree(allow_dirty: bool) -> dict[str, bool]:
        if not allow_dirty:
            raise SystemExit("dirty tree refused")
        return {"preregistration_violated": True}

    monkeypatch.setattr(run_stage3.manifests, "require_clean_tree", fake_require_clean_tree)

    with pytest.raises(SystemExit, match="dirty tree refused"):
        run_stage3.main(
            [
                "--profile",
                "smoke",
                "--out",
                str(tmp_path / "dirty_refused"),
                "--checkpoint-every",
                "1",
            ]
        )

    rc = run_stage3.main(
        [
            "--profile",
            "smoke",
            "--out",
            str(tmp_path / "dirty_allowed"),
            "--checkpoint-every",
            "1",
            "--allow-dirty",
        ]
    )
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "dirty_allowed" / "stage3_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    dirty_state = manifest["core"]["payload"]["dirty_state"]
    assert dirty_state["dirty_tree_allowed"] is True
    assert dirty_state["preregistration_violated"] is True


def test_stage3_mc_nc_columns_and_monotone_fixture() -> None:
    rows: list[dict[str, object]] = []
    for gamma in (0.1, 0.2, 0.3):
        for model in ("qrc", "qelm"):
            rows.extend(
                [
                    {
                        "gamma": gamma,
                        "dissipator": "local_loss",
                        "cell_id": "cell0",
                        "input_seed": 0,
                        "coupling_seed": 314159,
                        "shot_seed": 0,
                        "model": model,
                        "target": "leg_tau0_p1",
                        "target_family": "phase_diagram",
                        "family_cell": "tau0_p1",
                        "tau": 0,
                        "p": 1,
                        "capacity_test": 1.0 - gamma,
                    },
                    {
                        "gamma": gamma,
                        "dissipator": "local_loss",
                        "cell_id": "cell0",
                        "input_seed": 0,
                        "coupling_seed": 314159,
                        "shot_seed": 0,
                        "model": model,
                        "target": "leg_tau0_p2",
                        "target_family": "phase_diagram",
                        "family_cell": "tau0_p2",
                        "tau": 0,
                        "p": 2,
                        "capacity_test": gamma,
                    },
                ]
            )

    scored = run_stage3._attach_mc_nc_and_delta(pd.DataFrame(rows))
    assert {"MC", "NC"} <= set(scored.columns)
    summary = (
        scored.drop_duplicates(["gamma", "model"])
        .sort_values(["model", "gamma"])
        .groupby("model", sort=False)[["MC", "NC"]]
        .agg(list)
    )
    for row in summary.itertuples():
        assert row.MC == sorted(row.MC, reverse=True)
        assert row.NC == sorted(row.NC)


def test_stage3_t_evolve_is_fixed_across_gamma_for_coupling_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config()
    calls: list[float] = []

    def fake_recommend(cfg, coupling_seed):
        calls.append(float(cfg.gamma))
        t = 10.0 * float(cfg.gamma)
        return {
            "t_lo": 0.5 * t,
            "t_recommended": t,
            "t_hi": 2.0 * t,
            "regime": "dissipative",
            "timescale": t,
        }

    monkeypatch.setattr(run_stage3, "recommend_T_evolve", fake_recommend)

    values = []
    for gamma in config["gamma_grid"]:
        qelm_seed_cfg = run_stage3._fixed_configs(
            config,
            gamma=float(gamma),
            dissipator="local_loss",
        )["qelm_seed"]
        rec = run_stage3._stage3_t_evolve_recommendation(qelm_seed_cfg, 314159)
        values.append(float(rec["t_stage3"]))

    assert values == [3.0, 3.0]
    assert calls == [0.3, 0.3]


def test_stage3_qelm_capacity_mc_nc_vary_across_smoke_gamma_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    config["T"] = 72
    config["phase_diagram"]["degrees"] = [1, 2]
    config["phase_diagram"]["robustness_degrees"] = [1, 2]
    config["gamma_grid"] = [0.02, 0.3, 0.5]
    config["dissipator_grid"] = ["local_loss"]

    def reciprocal_recommend(cfg, coupling_seed):
        gamma = float(cfg.gamma)
        t = 1.0e9 if gamma == 0.0 else 0.3 / gamma
        return {
            "t_lo": 0.5 * t,
            "t_recommended": t,
            "t_hi": 2.0 * t,
            "regime": "dissipative",
            "timescale": t,
        }

    def fake_selected_or_cached_bundle(
        out_dir,
        model,
        item_id,
        cfg,
        inputs,
        coupling_seed,
        observables,
    ):
        n_obs = len(observables)
        if model == "qelm":
            marker = float(cfg.gamma) * float(cfg.T_evolve)
            features = _marker_features(len(inputs), n_obs, marker)
        else:
            features = _live_features(np.asarray(inputs), n_obs)
        bundle = run_stage1.FeatureBundle(
            features=features,
            model=model,
            config=asdict(cfg),
            couplings=np.zeros((int(cfg.n_qubits), int(cfg.n_qubits))),
            n_obs=n_obs,
            n_multiplex=1,
            first_valid=0,
        )
        path = run_stage1._feature_cache_path(out_dir, model, item_id)
        run_stage1._save_bundle(path, bundle)
        return bundle

    def fake_evaluate_feature_matrix(features, targets, splits, alphas, first_valid):
        marker = float(np.mean(np.asarray(features)[:, 0]))
        rows = []
        for target_name in targets:
            is_degree_two = str(target_name).endswith("_p2")
            rows.append(
                {
                    "target": target_name,
                    "capacity_test": marker**2 if is_degree_two else marker,
                    "nrmse_test": 1.0 - marker,
                    "alpha": 0.0,
                    "effective_rank": 1.0,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(run_stage3, "recommend_T_evolve", reciprocal_recommend)
    monkeypatch.setattr(
        run_stage3.run_stage1,
        "_selected_or_cached_bundle",
        fake_selected_or_cached_bundle,
    )
    monkeypatch.setattr(
        run_stage3.pipeline,
        "evaluate_feature_matrix",
        fake_evaluate_feature_matrix,
    )

    cell = SeedCell(input_seed=0, coupling_seed=314159)
    cells = [{"cell_id": run_stage1._cell_id(cell), "seed_cell": cell}]
    out_dir = tmp_path / "stage3_synthetic"
    manifest_state = {"cells": {}, "feature_cache": {}}
    frames = [
        run_stage3._score_gamma_item(
            item,
            config=config,
            out_dir=out_dir,
            manifest_state=manifest_state,
        )
        for item in run_stage3._stage3_items(cells, config)
    ]
    scores = pd.concat(frames, ignore_index=True)
    qelm = scores[scores["model"] == "qelm"]
    qelm_summary = qelm.drop_duplicates(["gamma", "model"]).sort_values("gamma")

    assert qelm.groupby("gamma")["capacity_test"].sum().nunique() == len(config["gamma_grid"])
    assert qelm_summary["MC"].nunique() == len(config["gamma_grid"])
    assert qelm_summary["NC"].nunique() == len(config["gamma_grid"])


def test_stage3_records_dead_features_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    config["phase_diagram"]["degrees"] = [1, 2]
    config["phase_diagram"]["robustness_degrees"] = [1, 2]

    monkeypatch.setattr(
        run_stage3,
        "recommend_T_evolve",
        lambda cfg, coupling_seed: {
            "t_lo": 0.5,
            "t_recommended": 1.0,
            "t_hi": 2.0,
            "regime": "dissipative",
            "timescale": 1.0,
        },
    )

    def fake_selected_or_cached_bundle(
        out_dir,
        model,
        item_id,
        cfg,
        inputs,
        coupling_seed,
        observables,
    ):
        n_obs = len(observables)
        features = (
            np.zeros((len(inputs), n_obs), dtype=float)
            if model == "qelm"
            else _live_features(np.asarray(inputs), n_obs)
        )
        bundle = run_stage1.FeatureBundle(
            features=features,
            model=model,
            config=asdict(cfg),
            couplings=np.zeros((int(cfg.n_qubits), int(cfg.n_qubits))),
            n_obs=n_obs,
            n_multiplex=1,
            first_valid=0,
        )
        path = run_stage1._feature_cache_path(out_dir, model, item_id)
        run_stage1._save_bundle(path, bundle)
        return bundle

    monkeypatch.setattr(
        run_stage3.run_stage1,
        "_selected_or_cached_bundle",
        fake_selected_or_cached_bundle,
    )

    cell = SeedCell(input_seed=0, coupling_seed=314159)
    item = run_stage3._stage3_items(
        [{"cell_id": run_stage1._cell_id(cell), "seed_cell": cell}],
        config,
    )[0]
    manifest_state = {"cells": {}, "feature_cache": {}}
    scores = run_stage3._score_gamma_item(
        item,
        config=config,
        out_dir=tmp_path / "dead",
        manifest_state=manifest_state,
    )
    qelm = scores[scores["model"] == "qelm"]

    assert qelm["features_dead"].eq(True).all()
    assert qelm["capacity_test"].eq(0.0).all()
    assert qelm["MC"].eq(0.0).all()
    assert qelm["NC"].eq(0.0).all()
    assert manifest_state["cells"][item["item_id"]]["features_dead"]["qelm"] is True
    assert "qelm" in manifest_state["cells"][item["item_id"]]["liveness_failures"]


def _marker_features(n_rows: int, n_cols: int, marker: float) -> np.ndarray:
    ramp = np.linspace(-0.5, 0.5, n_rows, dtype=float)
    columns = [float(marker) + 0.01 * ramp]
    for idx in range(1, n_cols):
        columns.append(np.sin((idx + 1.0 + float(marker)) * ramp) + 0.02 * idx * ramp)
    return np.column_stack(columns)


def _live_features(inputs: np.ndarray, n_cols: int) -> np.ndarray:
    x = 2.0 * np.asarray(inputs, dtype=float) - 1.0
    columns = [x, x**2 - float(np.mean(x**2))]
    for idx in range(2, n_cols):
        columns.append(np.sin((idx + 1.0) * inputs) + 0.05 * x)
    return np.column_stack(columns[:n_cols])
