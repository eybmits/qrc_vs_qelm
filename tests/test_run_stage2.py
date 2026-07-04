from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrc_vs_qelm import run_stage1, run_stage2
from qrc_vs_qelm.manifests import sha256_file, sha256_frame, write_manifest
from qrc_vs_qelm.seeds import SeedCell


def _tiny_config() -> dict:
    return {
        "profile": "smoke",
        "n_qubits": 1,
        "T": 48,
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
        "shot_tiers": ["infinity", 1000],
        "runtime_budget": {
            "stage_budget_hours": 10.0,
            "active_shrink": {"stage2": {"applied": False}},
        },
    }


def _write_stage1_fixture(stage1_out: Path, config: dict) -> None:
    stage1_out.mkdir(parents=True)
    (stage1_out / "features").mkdir()
    cell = SeedCell(input_seed=0, coupling_seed=314159, shot_seed=0)
    cell_id = run_stage1._cell_id(cell)
    rng = np.random.default_rng(123)
    inputs = rng.uniform(0.0, 1.0, size=int(config["T"]))
    qrc_features = np.column_stack(
        [
            0.8 * (2.0 * inputs - 1.0),
            0.6 * np.sin(inputs),
            0.5 * np.cos(inputs),
        ]
    )
    qelm_features = np.column_stack(
        [
            0.7 * (2.0 * inputs - 1.0),
            0.5 * inputs**2,
            0.4 * np.sin(2.0 * inputs),
        ]
    )
    qrc_bundle = run_stage1.FeatureBundle(
        features=qrc_features,
        model="qrc",
        config={
            "dt": 0.5,
            "h_in": 1.0,
            "n_virtual": 1,
            "n_qubits": 1,
            "J_s": 0.7,
            "h": 0.4,
            "gamma": 0.2,
            "dissipator": "local_loss",
            "quantize_levels": 5,
        },
        couplings=np.zeros((1, 1)),
        n_obs=3,
        n_multiplex=1,
        first_valid=0,
    )
    qelm_bundle = run_stage1.FeatureBundle(
        features=qelm_features,
        model="qelm",
        config={
            "d": 1,
            "T_evolve": 1.0,
            "n_snapshots": 1,
            "n_qubits": 1,
            "J_s": 0.7,
            "h": 0.4,
            "gamma": 0.2,
            "dissipator": "local_loss",
            "quantize_levels": 5,
        },
        couplings=np.zeros((1, 1)),
        n_obs=3,
        n_multiplex=1,
        first_valid=0,
    )
    qrc_path = run_stage1._feature_cache_path(stage1_out, "qrc", cell_id)
    qelm_path = run_stage1._feature_cache_path(stage1_out, "qelm", cell_id)
    run_stage1._save_bundle(qrc_path, qrc_bundle)
    run_stage1._save_bundle(qelm_path, qelm_bundle)
    write_manifest(
        stage1_out,
        "stage1",
        {
            "repo_root": run_stage1.ROOT,
            "stage": 1,
            "profile": "smoke",
            "config": config,
            "seed_grid": {
                "name": "stage1_smoke",
                "n_cells": 1,
                "cell_count": 1,
                "cells": [cell.__dict__],
            },
            "manifest_state": {
                "feature_cache": {
                    cell_id: {
                        "qrc": {"path": str(qrc_path), "sha256": sha256_file(qrc_path)},
                        "qelm": {"path": str(qelm_path), "sha256": sha256_file(qelm_path)},
                    }
                },
                "cells": {
                    cell_id: {
                        "selected": {
                            "qrc": qrc_bundle.config,
                            "qelm": qelm_bundle.config,
                        }
                    }
                },
            },
            "outputs": {"scores": str(stage1_out / "stage1_scores.csv")},
        },
    )


@pytest.fixture
def stage1_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict]:
    config = _tiny_config()
    stage1_out = tmp_path / "stage1"
    _write_stage1_fixture(stage1_out, config)
    monkeypatch.setattr(
        run_stage2.manifests,
        "require_clean_tree",
        lambda allow_dirty: {"preregistration_violated": False},
    )
    return stage1_out, config


def _run(stage1_out: Path, out: Path) -> pd.DataFrame:
    rc = run_stage2.main(
        [
            "--profile",
            "smoke",
            "--stage1-out",
            str(stage1_out),
            "--out",
            str(out),
            "--checkpoint-every",
            "1",
        ]
    )
    assert rc == 0
    return pd.read_csv(out / "stage2_scores.csv", float_precision="round_trip")


def test_stage2_smoke_main_writes_scores_and_manifest(stage1_fixture, tmp_path: Path) -> None:
    stage1_out, _ = stage1_fixture
    out = tmp_path / "stage2"
    scores = _run(stage1_out, out)
    manifest = json.loads((out / "stage2_manifest.json").read_text(encoding="utf-8"))
    payload = manifest["core"]["payload"]

    assert not scores.empty
    assert {"shots", "shot_seed"} <= set(scores.columns)
    assert set(scores["shots"].astype(str)) == {"1000", "∞"}
    assert {"qrc", "qelm", "linear_delay", "poly_delay"} <= set(scores["model"])
    assert payload["stage"] == 2
    assert payload["profile"] == "smoke"
    assert payload["outputs"]["n_score_rows"] == len(scores)
    assert payload["manifest_state"]["stage1_feature_cache"]
    assert (out / "checkpoints").is_dir()


def test_stage2_smoke_determinism(stage1_fixture, tmp_path: Path) -> None:
    stage1_out, _ = stage1_fixture
    left = _run(stage1_out, tmp_path / "left")
    right = _run(stage1_out, tmp_path / "right")

    assert sha256_frame(left) == sha256_frame(right)


def test_stage2_classical_controls_are_exact_across_shot_tiers(
    stage1_fixture,
    tmp_path: Path,
) -> None:
    stage1_out, _ = stage1_fixture
    scores = _run(stage1_out, tmp_path / "shot_controls")
    controls = scores[scores["model"].isin(["linear_delay", "poly_delay"])]

    assert not controls.empty
    for _, group in controls.groupby(
        ["model", "target", "input_seed", "coupling_seed"],
        sort=False,
    ):
        assert group["shots"].nunique() > 1
        for column in (
            "capacity_test",
            "nrmse_test",
            "alpha",
            "effective_rank",
            "effective_rank_diag",
            "multiplex_gain",
        ):
            assert group[column].nunique(dropna=False) == 1


def test_stage2_qrc_qelm_shot_noise_rng_draws_are_independent() -> None:
    features = np.zeros((16, 4), dtype=float)
    qrc = run_stage2.shots.add_shot_noise(
        features,
        1000,
        run_stage2._shot_noise_rng("qrc", tier_index=1, shot_seed=7),
    )
    qrc_repeat = run_stage2.shots.add_shot_noise(
        features,
        1000,
        run_stage2._shot_noise_rng("qrc", tier_index=1, shot_seed=7),
    )
    qelm = run_stage2.shots.add_shot_noise(
        features,
        1000,
        run_stage2._shot_noise_rng("qelm", tier_index=1, shot_seed=7),
    )

    assert np.array_equal(qrc, qrc_repeat)
    assert not np.array_equal(qrc, qelm)


def test_stage2_does_not_call_simulator_builders(
    stage1_fixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage1_out, _ = stage1_fixture

    def forbidden(*args, **kwargs):  # pragma: no cover - only runs on regression.
        raise AssertionError("Stage-2 must not build new simulator features")

    monkeypatch.setattr(run_stage2.run_stage1, "build_features", forbidden)
    _run(stage1_out, tmp_path / "guarded")


def test_stage2_source_has_no_simulator_calls() -> None:
    source = Path(run_stage2.__file__).read_text(encoding="utf-8")
    forbidden = [
        "build_features(",
        "build_qrc_features",
        "build_qelm_features",
        "LindbladReservoir",
        "reservoirs.",
        "qrc_vs_qelm.models.qelm",
    ]
    for needle in forbidden:
        assert needle not in source


def test_stage2_clean_tree_gate_blocks_dirty_without_allow(
    stage1_fixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage1_out, _ = stage1_fixture

    def fake_require_clean_tree(allow_dirty: bool) -> dict[str, bool]:
        if not allow_dirty:
            raise SystemExit("dirty tree refused")
        return {"preregistration_violated": True}

    monkeypatch.setattr(run_stage2.manifests, "require_clean_tree", fake_require_clean_tree)

    with pytest.raises(SystemExit, match="dirty tree refused"):
        run_stage2.main(
            [
                "--profile",
                "smoke",
                "--stage1-out",
                str(stage1_out),
                "--out",
                str(tmp_path / "dirty_refused"),
            ]
        )

    rc = run_stage2.main(
        [
            "--profile",
            "smoke",
            "--stage1-out",
            str(stage1_out),
            "--out",
            str(tmp_path / "dirty_allowed"),
            "--allow-dirty",
        ]
    )
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "dirty_allowed" / "stage2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    dirty_state = manifest["core"]["payload"]["dirty_state"]
    assert dirty_state["dirty_tree_allowed"] is True
    assert dirty_state["preregistration_violated"] is True
