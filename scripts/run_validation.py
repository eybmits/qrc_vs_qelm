#!/usr/bin/env python
"""WP6 validation gate for the QRC-vs-QELM comparison."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrc_vs_qelm import pipeline, task_families
from qrc_vs_qelm.core import operators, readout, reservoirs, sparse_evolve
from qrc_vs_qelm.matching import (
    assert_bundles_matched,
    assert_features_live,
    assert_matched,
    effective_dimension,
    multiplex_efficacy,
    physical_shot_budget,
    recommend_T_evolve,
)
from qrc_vs_qelm.models.api import QELMConfig, QRCConfig, build_features
from qrc_vs_qelm.seeds import SEED_GRIDS


RESULTS_DIR = ROOT / "results_validation"
SOURCE_RESULTS_DIR = Path("/Users/superposition/Coding/qrc_dissipation_engineering/results")
BEST_VS_BEST_PATH = SOURCE_RESULTS_DIR / "best_vs_best.json"
CONFIRM_STM_PATH = SOURCE_RESULTS_DIR / "confirm_stm.json"

ALPHAS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2)
STAGE1_COUPLING_SEEDS = tuple(
    dict.fromkeys(cell.coupling_seed for cell in SEED_GRIDS["stage1_full"])
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _target_tau(name: str) -> int:
    marker = "_tau"
    return int(name.split(marker, 1)[1].split("_", 1)[0])


def _qrc_cd_sparse_features(
    inputs: np.ndarray,
    *,
    n_qubits: int,
    coupling_seed: int,
    h: float,
    dt: float,
    gamma: float,
    j_scale: float,
    observables: list[readout.Observable],
) -> np.ndarray:
    rng = np.random.default_rng(int(coupling_seed))
    J = reservoirs.random_couplings(n_qubits, j_scale, rng)
    h0 = reservoirs.ising_xx_hamiltonian(J, h, n_qubits)
    hx = reservoirs.transverse_drive(n_qubits)
    jumps = [(operators.sminus(i, n_qubits), float(gamma)) for i in range(n_qubits)]
    return sparse_evolve.run_sparse_lindblad_trajectory(
        n_qubits=n_qubits,
        H_static=h0 + h * hx,
        H_drive=h * hx,
        jumps=jumps,
        dt=dt,
        inputs=inputs,
        observables=observables,
        washout=0,
        n_virtual=1,
    )


def _fn_features(
    inputs: np.ndarray,
    *,
    n_qubits: int,
    coupling_seed: int,
    h: float,
    dt: float,
    j_scale: float,
    observables: list[readout.Observable],
) -> np.ndarray:
    rng = np.random.default_rng(int(coupling_seed))
    J = reservoirs.random_couplings(n_qubits, j_scale, rng)
    reservoir = reservoirs.FujiNakajimaReservoir(n_qubits, J, h, dt)
    return reservoir.run(inputs, observables, washout=0, n_virtual=1)


def _score_feature_matrix(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    washout: int,
    max_delay: int,
    first_valid: int = 0,
) -> tuple[Any, pipeline.Splits]:
    idx = pipeline.valid_indices(
        features.shape[0],
        washout=washout,
        max_delay=max_delay,
        targets=targets,
    )
    splits = pipeline.make_splits(idx)
    scores = pipeline.evaluate_feature_matrix(
        features,
        targets,
        splits,
        alphas=ALPHAS,
        first_valid=first_valid,
    )
    return scores, splits


def check_v1_qrc_physics(profile: str) -> dict[str, Any]:
    best = _load_json(BEST_VS_BEST_PATH)
    confirm = _load_json(CONFIRM_STM_PATH)
    ref = best["results"]["CD_paper (uniform local)"]
    ref_mc = float(ref["stm_test_MC"])
    ref_mc_std = float(ref["stm_test_MC_std"])
    ref_nmse = float(ref["narma_test_NMSE"])
    ref_nmse_std = float(ref["narma_test_NMSE_std"])
    mc_band = [ref_mc - 2.0 * ref_mc_std, ref_mc + 2.0 * ref_mc_std]

    cfg = {
        "n_qubits": 5,
        "h": 0.5,
        "dt": 0.5,
        "gamma": 0.3,
        "j_scale": 1.0,
        "input_seed": 0,
        "coupling_seeds": list(STAGE1_COUPLING_SEEDS),
        "delays": list(range(1, 16)),
    }
    if profile == "smoke":
        cfg.update(
            {
                "T": 360,
                "washout": 60,
                "criterion": "relaxed_smoke_directional",
                "note": (
                    "Smoke keeps N=5 and the stage1 coupling seeds, but uses "
                    "shorter sequences; it records the strict source band and "
                    "passes on directional V1 plus V2/V3."
                ),
            }
        )
    else:
        cfg.update({"T": 1200, "washout": 200, "criterion": "strict_reference_band"})

    observables = readout.pauli_observables(cfg["n_qubits"], axes=("x", "y", "z"), max_weight=2)
    inputs = np.random.default_rng(cfg["input_seed"]).uniform(0.0, 1.0, size=cfg["T"])
    stm_targets = task_families.stm_targets(inputs, cfg["delays"])
    narma10 = task_families.narma_targets(inputs)["narma10"]
    all_targets = {**stm_targets, "narma10": narma10}

    per_seed: list[dict[str, Any]] = []
    for seed in cfg["coupling_seeds"]:
        cd_features = _qrc_cd_sparse_features(
            inputs,
            n_qubits=cfg["n_qubits"],
            coupling_seed=seed,
            h=cfg["h"],
            dt=cfg["dt"],
            gamma=cfg["gamma"],
            j_scale=cfg["j_scale"],
            observables=observables,
        )
        cd_scores, splits = _score_feature_matrix(
            cd_features,
            all_targets,
            washout=cfg["washout"],
            max_delay=max(cfg["delays"]),
        )
        stm_scores = cd_scores[cd_scores["target"].str.startswith("stm_tau")].copy()
        stm_scores["tau"] = stm_scores["target"].map(_target_tau)
        narma_row = cd_scores[cd_scores["target"] == "narma10"].iloc[0]

        fn_features = _fn_features(
            inputs,
            n_qubits=cfg["n_qubits"],
            coupling_seed=seed,
            h=cfg["h"],
            dt=cfg["dt"],
            j_scale=cfg["j_scale"],
            observables=observables,
        )
        fn_scores, _ = _score_feature_matrix(
            fn_features,
            stm_targets,
            washout=cfg["washout"],
            max_delay=max(cfg["delays"]),
        )
        fn_scores["tau"] = fn_scores["target"].map(_target_tau)

        cd_short = stm_scores[stm_scores["tau"] <= 5]
        fn_short = fn_scores[fn_scores["tau"] <= 5]
        per_seed.append(
            {
                "coupling_seed": int(seed),
                "stm_MC": float(stm_scores["capacity_test"].sum()),
                "narma10_NMSE": float(narma_row["nrmse_test"]) ** 2,
                "cd_short_delay_capacity_mean": float(cd_short["capacity_test"].mean()),
                "fn_short_delay_capacity_mean": float(fn_short["capacity_test"].mean()),
                "split_rows": {
                    "train": int(splits.train.size),
                    "val": int(splits.val.size),
                    "test": int(splits.test.size),
                },
            }
        )

    mc_values = np.array([row["stm_MC"] for row in per_seed], dtype=float)
    nmse_values = np.array([row["narma10_NMSE"] for row in per_seed], dtype=float)
    cd_short_values = np.array(
        [row["cd_short_delay_capacity_mean"] for row in per_seed], dtype=float
    )
    fn_short_values = np.array(
        [row["fn_short_delay_capacity_mean"] for row in per_seed], dtype=float
    )

    strict_stm_pass = bool(mc_band[0] <= float(mc_values.mean()) <= mc_band[1])
    smoke_stm_threshold = 0.5 * ref_mc
    stm_pass = strict_stm_pass
    if profile == "smoke":
        stm_pass = bool(float(mc_values.mean()) >= smoke_stm_threshold)
    narma_pass = bool(float(nmse_values.mean()) < 0.7)
    ordering_pass = bool(float(cd_short_values.mean()) >= float(fn_short_values.mean()))

    confirm_fn = np.asarray(confirm["cap_fn"], dtype=float)
    confirm_cd = np.asarray(confirm["cap_cd"], dtype=float)
    short_mask = np.asarray(confirm["delays"], dtype=int) <= 5

    checks = {
        "stm_MC": {
            "measured_mean": float(mc_values.mean()),
            "measured_std": float(mc_values.std(ddof=0)),
            "reference_mean": ref_mc,
            "reference_std": ref_mc_std,
            "tolerance_band": mc_band,
            "strict_pass": strict_stm_pass,
            "pass": stm_pass,
            "source_file": str(BEST_VS_BEST_PATH),
            "source_json_pointer": "/results/CD_paper (uniform local)/stm_test_MC",
        },
        "narma10_NMSE": {
            "measured_mean": float(nmse_values.mean()),
            "measured_std": float(nmse_values.std(ddof=0)),
            "reference_mean": ref_nmse,
            "reference_std": ref_nmse_std,
            "directional_bound": 0.7,
            "pass": narma_pass,
            "source_file": str(BEST_VS_BEST_PATH),
            "source_json_pointer": "/results/CD_paper (uniform local)/narma_test_NMSE",
        },
        "short_delay_ordering": {
            "measured_cd_mean_tau_le_5": float(cd_short_values.mean()),
            "measured_fn_mean_tau_le_5": float(fn_short_values.mean()),
            "reference_cd_mean_tau_le_5": float(confirm_cd[:, short_mask].mean()),
            "reference_fn_mean_tau_le_5": float(confirm_fn[:, short_mask].mean()),
            "pass": ordering_pass,
            "source_file": str(CONFIRM_STM_PATH),
            "source_json_pointer": "/cap_cd and /cap_fn at delays <= 5",
        },
    }
    return {
        "passed": bool(stm_pass and narma_pass and ordering_pass),
        "profile": profile,
        "config": cfg,
        "checks": checks,
        "per_seed": per_seed,
    }


def check_v2_qelm_sanity() -> dict[str, Any]:
    n_qubits = 5
    d = 5
    coupling_seed = 314159
    inputs = np.random.default_rng(20240617).uniform(0.02, 0.98, size=360)
    base = QELMConfig(
        d=d,
        T_evolve=0.1,
        n_snapshots=1,
        n_qubits=n_qubits,
        J_s=1.0,
        h=0.5,
        gamma=0.0,
        dissipator="local_loss",
        quantize_levels=None,
    )
    recommendation = recommend_T_evolve(base, coupling_seed)
    cfg = replace(base, T_evolve=float(recommendation["t_lo"]))
    bundle = build_features(cfg, inputs, coupling_seed)

    live_pass = True
    live_error = None
    try:
        assert_features_live(bundle)
    except Exception as exc:  # pragma: no cover - reported in JSON
        live_pass = False
        live_error = str(exc)

    taus = list(range(0, 11))
    targets = task_families.build_family_targets(inputs, taus=taus, degrees=[1])
    idx = pipeline.stage_valid_indices(
        [bundle],
        targets,
        washout=10,
        max_delay=max(taus),
    )
    splits = pipeline.make_splits(idx)
    scores = pipeline.evaluate_feature_matrix(
        bundle.features,
        targets,
        splits,
        alphas=ALPHAS,
        first_valid=bundle.first_valid,
    )
    capacities = {
        _target_tau(row["target"]): float(row["capacity_test"])
        for _, row in scores.iterrows()
    }
    seen_taus = list(range(d))
    margin = 2
    unseen_taus = [tau for tau in taus if tau >= d + margin]
    seen_min = min(capacities[tau] for tau in seen_taus)
    unseen_max = max(capacities[tau] for tau in unseen_taus)
    seen_pass = bool(seen_min >= 0.99)
    unseen_pass = bool(unseen_max <= 0.05)
    diag = effective_dimension(bundle.features[bundle.first_valid :])
    gain = multiplex_efficacy(
        bundle.features[bundle.first_valid :],
        bundle.n_multiplex,
        bundle.n_obs,
    )

    return {
        "passed": bool(seen_pass and unseen_pass and live_pass),
        "config": {**bundle.config, "coupling_seed": coupling_seed},
        "recommendation": recommendation,
        "selected_T_evolve": float(cfg.T_evolve),
        "checks": {
            "visible_window_tau_lt_d": {
                "taus": seen_taus,
                "min_capacity": float(seen_min),
                "threshold": 0.99,
                "pass": seen_pass,
            },
            "cutoff_tau_ge_d_plus_margin": {
                "taus": unseen_taus,
                "max_capacity": float(unseen_max),
                "threshold": 0.05,
                "margin": margin,
                "pass": unseen_pass,
            },
            "features_live": {
                "pass": live_pass,
                "error": live_error,
                "effective_dimension": diag,
                "multiplex": gain,
            },
        },
        "capacities_by_tau": capacities,
    }


def check_v3_matched_budget() -> dict[str, Any]:
    n_qubits = 5
    coupling_seed = 314159
    shots_per_expectation = 10_000
    inputs = np.random.default_rng(20240704).uniform(0.02, 0.98, size=24)
    qrc_cfg = QRCConfig(
        dt=0.5,
        h_in=0.5,
        n_virtual=1,
        n_qubits=n_qubits,
        J_s=1.0,
        h=0.5,
        gamma=0.3,
        dissipator="local_loss",
        quantize_levels=None,
    )
    qelm_base = QELMConfig(
        d=n_qubits,
        T_evolve=0.1,
        n_snapshots=1,
        n_qubits=n_qubits,
        J_s=1.0,
        h=0.5,
        gamma=0.3,
        dissipator="local_loss",
        quantize_levels=None,
    )
    recommendation = recommend_T_evolve(qelm_base, coupling_seed)
    qelm_cfg = replace(qelm_base, T_evolve=float(recommendation["t_recommended"]))

    check_results: dict[str, Any] = {}
    passed = True
    try:
        assert_matched(qrc_cfg, qelm_cfg)
        check_results["config_assert_matched"] = {"pass": True}
    except Exception as exc:
        check_results["config_assert_matched"] = {"pass": False, "error": str(exc)}
        passed = False

    qrc_bundle = build_features(qrc_cfg, inputs, coupling_seed)
    qelm_bundle = build_features(qelm_cfg, inputs, coupling_seed)

    try:
        assert_bundles_matched(qrc_bundle, qelm_bundle)
        check_results["bundle_assert_matched"] = {"pass": True}
    except Exception as exc:
        check_results["bundle_assert_matched"] = {"pass": False, "error": str(exc)}
        passed = False

    budgets = {}
    for bundle in (qrc_bundle, qelm_bundle):
        budget = physical_shot_budget(
            shots_per_expectation,
            n_timesteps=bundle.features.shape[0],
            readout_dim=bundle.features.shape[1],
        )
        budgets[bundle.model] = int(budget)
    budget_pass = bool(budgets["qrc"] == budgets["qelm"])
    passed = passed and budget_pass
    check_results["physical_shot_budget"] = {
        "pass": budget_pass,
        "shots_per_expectation": shots_per_expectation,
        "budgets": budgets,
    }

    diagnostics: dict[str, Any] = {}
    for bundle in (qrc_bundle, qelm_bundle):
        live_pass = True
        live_error = None
        try:
            assert_features_live(bundle)
        except Exception as exc:
            live_pass = False
            live_error = str(exc)
            passed = False
        valid_features = bundle.features[bundle.first_valid :]
        diagnostics[bundle.model] = {
            "features_live": {"pass": live_pass, "error": live_error},
            "effective_dimension": effective_dimension(valid_features),
            "multiplex": multiplex_efficacy(
                valid_features,
                bundle.n_multiplex,
                bundle.n_obs,
            ),
            "feature_shape": list(bundle.features.shape),
            "first_valid": int(bundle.first_valid),
            "n_obs": int(bundle.n_obs),
            "n_multiplex": int(bundle.n_multiplex),
        }
    check_results["features_live"] = {
        "pass": all(d["features_live"]["pass"] for d in diagnostics.values())
    }

    return {
        "passed": bool(passed),
        "config": {
            "qrc": qrc_bundle.config,
            "qelm": qelm_bundle.config,
            "coupling_seed": coupling_seed,
            "qelm_recommendation": recommendation,
        },
        "checks": check_results,
        "diagnostics": diagnostics,
    }


def run_validation(profile: str) -> dict[str, Any]:
    checks = {
        "V1_qrc_physics_reproduction": check_v1_qrc_physics(profile),
        "V2_qelm_sanity": check_v2_qelm_sanity(),
        "V3_matched_budget_audit": check_v3_matched_budget(),
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "profile": profile,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args(argv)

    result = run_validation(args.profile)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "validation.json"
    out_path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
