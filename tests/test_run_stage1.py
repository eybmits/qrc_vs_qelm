from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrc_vs_qelm import run_stage1
from qrc_vs_qelm.manifests import sha256_frame
from qrc_vs_qelm.seeds import SeedCell


def _tiny_config() -> dict:
    return {
        "profile": "smoke",
        "n_qubits": 1,
        "T": 36,
        "washout": 5,
        "primary": {
            "n_qubits": 1,
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
                "dimension": 3,
            },
        },
        "phase_diagram": {
            "family": "legendre_delay",
            "taus": [0, 1],
            "degrees": [1],
            "gated_cells": 2,
            "robustness_degrees": [1],
            "robustness_cells": 2,
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
        "runtime_budget": {
            "stage_budget_hours": 10.0,
            "active_shrink": {"stage1": {"applied": False}},
        },
    }


@pytest.fixture
def tiny_stage1(monkeypatch: pytest.MonkeyPatch) -> dict:
    config = _tiny_config()
    monkeypatch.setattr(run_stage1, "_load_config", lambda profile: copy.deepcopy(config))
    monkeypatch.setattr(
        run_stage1.manifests,
        "require_clean_tree",
        lambda allow_dirty: {"preregistration_violated": False},
    )
    return config


def _run(out: Path, extra_args: list[str] | None = None) -> pd.DataFrame:
    args = ["--profile", "smoke", "--out", str(out), "--checkpoint-every", "1"]
    if extra_args:
        args.extend(extra_args)
    rc = run_stage1.main(args)
    assert rc == 0
    return pd.read_csv(out / "stage1_scores.csv", float_precision="round_trip")


def test_stage1_smoke_main_writes_scores_and_manifest(tiny_stage1, tmp_path: Path) -> None:
    out = tmp_path / "stage1"
    scores = _run(out)
    manifest = json.loads((out / "stage1_manifest.json").read_text(encoding="utf-8"))
    payload = manifest["core"]["payload"]

    assert not scores.empty
    assert set(run_stage1.SCORE_COLUMNS) <= set(scores.columns)
    assert {"qrc", "qelm", "linear_delay", "poly_delay"} <= set(scores["model"])
    assert scores["cell_id"].nunique() == 1
    assert scores["target"].nunique() == 2
    assert payload["stage"] == 1
    assert payload["profile"] == "smoke"
    assert payload["seed_grid"]["name"] == "stage1_smoke"
    assert payload["outputs"]["n_score_rows"] == len(scores)
    assert payload["manifest_state"]["gates"]["fdr_family_observed_cells"] == 2
    cell_id = scores["cell_id"].iloc[0]
    assert set(payload["manifest_state"]["feature_cache"][cell_id]) == {"qrc", "qelm"}
    assert (out / "checkpoints").is_dir()


def test_stage1_smoke_determinism(tiny_stage1, tmp_path: Path) -> None:
    left = _run(tmp_path / "left")
    right = _run(tmp_path / "right")

    assert sha256_frame(left) == sha256_frame(right)


def test_stage1_resumes_from_partial_checkpoint(
    tiny_stage1,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cells = (
        SeedCell(input_seed=0, coupling_seed=314159),
        SeedCell(input_seed=1, coupling_seed=271828),
    )
    monkeypatch.setattr(run_stage1, "SEED_GRIDS", {"stage1_smoke": cells})
    monkeypatch.setattr(
        run_stage1,
        "grid_manifest",
        lambda name: {
            "name": name,
            "n_cells": len(cells),
            "cell_count": len(cells),
            "cells": [cell.__dict__ for cell in cells],
        },
    )

    real_score = run_stage1._score_seed_cell
    calls = {"n": 0}

    def flaky_score(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_score(*args, **kwargs)

    out = tmp_path / "resumed"
    monkeypatch.setattr(run_stage1, "_score_seed_cell", flaky_score)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_stage1.main(
            ["--profile", "smoke", "--out", str(out), "--checkpoint-every", "1"]
        )

    assert (out / "checkpoints" / "stage1_scores_chunk_0001.csv").exists()

    monkeypatch.setattr(run_stage1, "_score_seed_cell", real_score)
    resumed = _run(out)
    fresh = _run(tmp_path / "fresh")

    assert sha256_frame(resumed) == sha256_frame(fresh)
    manifest = json.loads((out / "stage1_manifest.json").read_text(encoding="utf-8"))
    cache = manifest["core"]["payload"]["manifest_state"]["feature_cache"]
    assert set(cache) == {run_stage1._cell_id(cell) for cell in cells}


def test_stage1_mismatched_config_raises(
    tiny_stage1,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad = copy.deepcopy(tiny_stage1)
    bad["primary"]["n_snapshots"] = 2
    monkeypatch.setattr(run_stage1, "_load_config", lambda profile: copy.deepcopy(bad))

    with pytest.raises(Exception, match="matched-resource|bundle-resource|feature"):
        run_stage1.main(
            ["--profile", "smoke", "--out", str(tmp_path / "bad"), "--checkpoint-every", "1"]
        )


def test_stage1_clean_tree_gate_blocks_dirty_without_allow(
    tiny_stage1,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_require_clean_tree(allow_dirty: bool) -> dict[str, bool]:
        if not allow_dirty:
            raise SystemExit("dirty tree refused")
        return {"preregistration_violated": True}

    monkeypatch.setattr(
        run_stage1.manifests,
        "require_clean_tree",
        fake_require_clean_tree,
    )

    with pytest.raises(SystemExit, match="dirty tree refused"):
        run_stage1.main(
            [
                "--profile",
                "smoke",
                "--out",
                str(tmp_path / "dirty_refused"),
                "--checkpoint-every",
                "1",
            ]
        )

    _run(tmp_path / "dirty_allowed", extra_args=["--allow-dirty"])
    manifest = json.loads(
        (tmp_path / "dirty_allowed" / "stage1_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    dirty_state = manifest["core"]["payload"]["dirty_state"]
    assert dirty_state["dirty_tree_allowed"] is True
    assert dirty_state["preregistration_violated"] is True


def _synthetic_gate_scores() -> pd.DataFrame:
    specs = [
        {
            "family_cell": "tau0_p2",
            "tau": 0,
            "p": 2,
            "qrc": [0.04] * 6,
            "qelm": [0.005] * 6,
            "linear_delay": [0.02] * 6,
            "poly_delay": [0.01] * 6,
        },
        {
            "family_cell": "tau1_p1",
            "tau": 1,
            "p": 1,
            "qrc": [0.04] * 6,
            "qelm": [0.12] * 6,
            "linear_delay": [0.02] * 6,
            "poly_delay": [0.02] * 6,
        },
        {
            "family_cell": "tau2_p3",
            "tau": 2,
            "p": 3,
            "qrc": [0.04, 0.04, 0.04, 0.04, 0.04, 0.11],
            "qelm": [0.01] * 6,
            "linear_delay": [0.0] * 6,
            "poly_delay": [0.0] * 6,
        },
    ]
    rows: list[dict[str, object]] = []
    for seed_idx in range(6):
        for spec in specs:
            for model in ("qrc", "qelm", "linear_delay", "poly_delay"):
                rows.append(
                    {
                        "cell_id": f"seed{seed_idx}",
                        "input_seed": seed_idx,
                        "coupling_seed": 314159,
                        "shot_seed": 0,
                        "model": model,
                        "target": spec["family_cell"],
                        "target_family": "phase_diagram",
                        "family_cell": spec["family_cell"],
                        "confirmatory": True,
                        "tau": spec["tau"],
                        "p": spec["p"],
                        "capacity_test": spec[model][seed_idx],
                        "nrmse_test": 0.0,
                        "alpha": 0.0,
                        "effective_rank": 1.0,
                        "effective_rank_diag": 1.0,
                        "multiplex_gain": 1.0,
                        "D": 3,
                        "n_obs": 3,
                        "n_multiplex": 1,
                        "selected_hp": "{}",
                        "gamma": 0.2,
                        "uninformative": False,
                        "capacity_linear_delay": np.nan,
                        "capacity_poly_delay": np.nan,
                        "complete": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_uninformative_exclusion_is_family_cell_mean_and_reported(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    config["phase_diagram"]["gated_cells"] = 3
    scores = run_stage1._mark_uninformative(_synthetic_gate_scores(), config)
    payload = run_stage1._stage1_gate_payload(scores, config)

    assert payload["uninformative_cells"] == ["tau0_p2"]
    assert sorted(payload["informative_cells"]) == ["tau1_p1", "tau2_p3"]

    csv_path = tmp_path / "stage1_scores.csv"
    run_stage1._write_scores(scores, csv_path)
    csv_scores = pd.read_csv(csv_path, float_precision="round_trip")
    assert set(csv_scores["family_cell"]) == {"tau0_p2", "tau1_p1", "tau2_p3"}
    assert csv_scores.loc[csv_scores["family_cell"] == "tau0_p2", "uninformative"].all()
    assert not csv_scores.loc[
        csv_scores["family_cell"] != "tau0_p2", "uninformative"
    ].any()

    manifest_uninformative = set(payload["uninformative_cells"])
    manifest_informative = set(payload["informative_cells"])
    assert manifest_uninformative.isdisjoint(manifest_informative)
    for row in csv_scores.itertuples(index=False):
        expected = row.family_cell in manifest_uninformative
        assert bool(row.uninformative) is expected

    excluded = {
        item["family_cell"]: item for item in payload["uninformative_cell_capacities"]
    }
    assert set(excluded) == {"tau0_p2"}
    assert excluded["tau0_p2"]["capacity_qrc_mean"] == pytest.approx(0.04)
    assert excluded["tau0_p2"]["capacity_qelm_mean"] == pytest.approx(0.005)

    with_exclusion = payload["g1b_fraction_with_exclusion"]
    without_exclusion = payload["g1b_fraction_without_exclusion"]
    assert with_exclusion["n_cells"] == 2
    assert without_exclusion["n_cells"] == 3
    assert with_exclusion["qrc_fraction"] != without_exclusion["qrc_fraction"]
    assert with_exclusion["qelm_fraction"] != without_exclusion["qelm_fraction"]

    assert payload["g1c_fraction_with_exclusion"] == pytest.approx(1.0)
    assert payload["g1c_fraction_without_exclusion"] == pytest.approx(0.5)
    assert payload["records_full_unfiltered"]


def test_lucky_seed_family_cell_uses_mean_not_majority_seed_rule() -> None:
    config = _tiny_config()
    config["phase_diagram"]["gated_cells"] = 3
    scores = run_stage1._mark_uninformative(_synthetic_gate_scores(), config)
    lucky = scores[scores["family_cell"] == "tau2_p3"]

    qrc_mean = lucky.loc[lucky["model"] == "qrc", "capacity_test"].mean()
    assert qrc_mean > 0.05
    assert (lucky.loc[lucky["model"] == "qrc", "capacity_test"] < 0.05).sum() == 5
    assert not lucky["uninformative"].any()


def test_stage1_bundle_level_gate_catches_built_bundle_mismatch(
    tiny_stage1,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_build_features = run_stage1.build_features

    def mismatched_build_features(cfg, inputs, coupling_seed, observables=None):
        bundle = real_build_features(
            cfg,
            inputs,
            coupling_seed,
            observables=observables,
        )
        if isinstance(cfg, run_stage1.QELMConfig):
            return run_stage1.FeatureBundle(
                features=np.asarray(bundle.features)[:, :1],
                model=bundle.model,
                config=bundle.config,
                couplings=bundle.couplings,
                n_obs=bundle.n_obs,
                n_multiplex=bundle.n_multiplex,
                first_valid=bundle.first_valid,
            )
        return bundle

    monkeypatch.setattr(run_stage1, "build_features", mismatched_build_features)

    with pytest.raises(Exception, match="bundle-resource"):
        run_stage1.main(
            [
                "--profile",
                "smoke",
                "--out",
                str(tmp_path / "bundle_mismatch"),
                "--checkpoint-every",
                "1",
            ]
        )
