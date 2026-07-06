from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrc_vs_qelm import run_stage1, run_stage_ipc
from qrc_vs_qelm.manifests import sha256_frame
from qrc_vs_qelm.models.api import FeatureBundle
from qrc_vs_qelm.seeds import SeedCell


def _tiny_config() -> dict:
    return {
        "profile": "smoke",
        "n_qubits": 2,
        "T": 80,
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
                "axes": ["x", "z"],
                "max_weight": 1,
                "dimension": 4,
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
            "alpha": [0.0, 1.0e-8],
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
        "ipc": {
            "window_grid": [1, 2, 3],
            "basis": {
                "max_delay": 2,
                "max_degree": 1,
                "max_total_degree": 1,
                "max_functions": 3,
            },
            "n_surrogates": 2,
        },
        "runtime_budget": {
            "stage_budget_hours": 10.0,
            "active_shrink": {"stage_ipc": {"applied": False}},
        },
    }


@pytest.fixture
def tiny_ipc(monkeypatch: pytest.MonkeyPatch) -> dict:
    config = _tiny_config()
    cells = (SeedCell(input_seed=0, coupling_seed=314159),)
    monkeypatch.setattr(
        run_stage_ipc.run_stage1,
        "_load_config",
        lambda profile: copy.deepcopy(config),
    )
    monkeypatch.setattr(
        run_stage_ipc.manifests,
        "require_clean_tree",
        lambda allow_dirty: {"preregistration_violated": False},
    )
    monkeypatch.setattr(run_stage_ipc, "SEED_GRIDS", {"stage3_gamma": cells})
    monkeypatch.setattr(
        run_stage_ipc,
        "grid_manifest",
        lambda name: {
            "name": name,
            "n_cells": len(cells),
            "cell_count": len(cells),
            "cells": [cell.__dict__ for cell in cells],
        },
    )
    monkeypatch.setattr(
        run_stage_ipc.run_stage1,
        "_selected_or_cached_bundle",
        _fake_selected_or_cached_bundle,
    )
    return config


def _run(out: Path) -> pd.DataFrame:
    rc = run_stage_ipc.main(
        [
            "--profile",
            "smoke",
            "--out",
            str(out),
            "--checkpoint-every",
            "2",
            "--allow-dirty",
        ]
    )
    assert rc == 0
    return pd.read_csv(out / "ipc_scores.csv", float_precision="round_trip")


def test_ipc_smoke_main_writes_scores_and_manifest(
    tiny_ipc,
    tmp_path: Path,
) -> None:
    out = tmp_path / "ipc"
    scores = _run(out)
    manifest = json.loads((out / "ipc_manifest.json").read_text(encoding="utf-8"))
    payload = manifest["core"]["payload"]

    assert not scores.empty
    assert set(run_stage_ipc.IPC_SCORE_COLUMNS) == set(scores.columns)
    assert {"qrc", "qelm"} == set(scores["model"])
    assert set(scores["window_k"]) == {1, 2, 3}
    assert payload["stage"] == "ipc"
    assert payload["profile"] == "smoke"
    assert payload["outputs"]["n_score_rows"] == len(scores)
    assert payload["manifest_state"]["feature_cache"]
    assert (out / "checkpoints").is_dir()


def test_ipc_smoke_determinism(tiny_ipc, tmp_path: Path) -> None:
    left = _run(tmp_path / "left")
    right = _run(tmp_path / "right")

    assert sha256_frame(left) == sha256_frame(right)


def test_ipc_qrc_ceiling_and_nonnegative_delta_on_smoke(
    tiny_ipc,
    tmp_path: Path,
) -> None:
    scores = _run(tmp_path / "ceiling")
    pivot = scores.pivot_table(
        index=["gamma", "dissipator", "cell_id", "window_k"],
        columns="model",
        values="ipc_total",
        aggfunc="first",
    )

    assert (pivot["qrc"] + 1.0e-9 >= pivot["qelm"]).all()
    assert (scores["delta_ipc"] >= -1.0e-9).all()


def test_ipc_qelm_capacity_is_monotone_on_average(
    tiny_ipc,
    tmp_path: Path,
) -> None:
    scores = _run(tmp_path / "monotone")
    qelm_means = (
        scores[scores["model"] == "qelm"]
        .groupby("window_k", sort=True)["ipc_total"]
        .mean()
        .to_numpy()
    )

    assert np.all(np.diff(qelm_means) >= -0.05)


def test_ipc_clean_tree_gate_blocks_dirty_without_allow(
    tiny_ipc,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_require_clean_tree(allow_dirty: bool) -> dict[str, bool]:
        if not allow_dirty:
            raise SystemExit("dirty tree refused")
        return {"preregistration_violated": True}

    monkeypatch.setattr(
        run_stage_ipc.manifests,
        "require_clean_tree",
        fake_require_clean_tree,
    )

    with pytest.raises(SystemExit, match="dirty tree refused"):
        run_stage_ipc.main(
            [
                "--profile",
                "smoke",
                "--out",
                str(tmp_path / "dirty_refused"),
                "--checkpoint-every",
                "1",
            ]
        )

    rc = run_stage_ipc.main(
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
        (tmp_path / "dirty_allowed" / "ipc_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    dirty_state = manifest["core"]["payload"]["dirty_state"]
    assert dirty_state["dirty_tree_allowed"] is True
    assert dirty_state["preregistration_violated"] is True


def test_ipc_records_dead_features_without_crashing(
    tiny_ipc,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_dead_qelm_bundle(
        out_dir,
        model,
        item_id,
        cfg,
        inputs,
        coupling_seed,
        observables,
    ):
        bundle = _fake_selected_or_cached_bundle(
            out_dir,
            model,
            item_id,
            cfg,
            inputs,
            coupling_seed,
            observables,
        )
        if model == "qelm":
            bundle.features[:] = 0.0
            path = run_stage1._feature_cache_path(out_dir, model, item_id)
            run_stage1._save_bundle(path, bundle)
        return bundle

    monkeypatch.setattr(
        run_stage_ipc.run_stage1,
        "_selected_or_cached_bundle",
        fake_dead_qelm_bundle,
    )
    cell = SeedCell(input_seed=0, coupling_seed=314159)
    cells = [{"cell_id": run_stage1._cell_id(cell), "seed_cell": cell}]
    item = run_stage_ipc._ipc_items(cells, tiny_ipc)[0]
    manifest_state = {"cells": {}, "feature_cache": {}, "qrc_ipc_cache": {}}
    scores = run_stage_ipc._score_ipc_item(
        item,
        config=tiny_ipc,
        out_dir=tmp_path / "dead",
        manifest_state=manifest_state,
    )
    qelm = scores[scores["model"] == "qelm"]

    assert qelm["features_dead"].eq(True).all()
    assert qelm["ipc_total"].eq(0.0).all()
    assert qelm["ipc_mc"].eq(0.0).all()
    assert qelm["ipc_nc"].eq(0.0).all()
    assert manifest_state["cells"][item["item_id"]]["features_dead"]["qelm"] is True
    assert "qelm" in manifest_state["cells"][item["item_id"]]["liveness_failures"]


def test_ipc_shared_valid_indices_are_identical_across_k() -> None:
    T = 24
    targets = {"y": np.arange(T, dtype=float)}
    features = np.ones((T, 3), dtype=float)
    qrc = FeatureBundle(
        features=features,
        model="qrc",
        config={},
        couplings=np.zeros((2, 2)),
        n_obs=3,
        n_multiplex=1,
        first_valid=0,
    )

    split_rows = []
    for k in (1, 2, 3):
        qelm = FeatureBundle(
            features=features,
            model="qelm",
            config={"d": k},
            couplings=np.zeros((2, 2)),
            n_obs=3,
            n_multiplex=1,
            first_valid=k - 1,
        )
        splits = run_stage_ipc._shared_ipc_splits(
            qrc,
            qelm,
            targets=targets,
            washout=0,
            basis_max_delay=0,
            max_window_k=3,
        )
        split_rows.append(splits.all)

    assert all(np.array_equal(split_rows[0], rows) for rows in split_rows[1:])
    assert split_rows[0][0] == 2


def _fake_selected_or_cached_bundle(
    out_dir,
    model,
    item_id,
    cfg,
    inputs,
    coupling_seed,
    observables,
):
    n_obs = len(observables)
    if model == "qrc":
        d = 3
        n_multiplex = int(cfg.n_virtual)
        first_valid = 0
    else:
        d = int(cfg.d)
        n_multiplex = int(cfg.n_snapshots)
        first_valid = d - 1

    D = n_obs * n_multiplex
    features = _controlled_delay_features(np.asarray(inputs), d=d, D=D)
    bundle = FeatureBundle(
        features=features,
        model=model,
        config=asdict(cfg),
        couplings=np.zeros((int(cfg.n_qubits), int(cfg.n_qubits))),
        n_obs=n_obs,
        n_multiplex=n_multiplex,
        first_valid=first_valid,
    )
    path = run_stage1._feature_cache_path(out_dir, model, item_id)
    run_stage1._save_bundle(path, bundle)
    return bundle


def _controlled_delay_features(inputs: np.ndarray, *, d: int, D: int) -> np.ndarray:
    values = np.asarray(inputs, dtype=np.float64)
    features = np.empty((values.shape[0], D), dtype=np.float64)
    t = np.linspace(-1.0, 1.0, values.shape[0], dtype=np.float64)
    for col in range(D):
        if col < d:
            if col == 0:
                features[:, col] = values
            else:
                features[:col, col] = values[0]
                features[col:, col] = values[:-col]
        else:
            features[:, col] = np.sin((col + 2.0) * np.pi * t) + 0.1 * np.cos(
                (col + 3.0) * np.pi * t
            )
    return features
