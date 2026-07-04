"""Stage-1 runner for the pre-registered RQ1 phase diagram."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from qrc_vs_qelm import controls, gates, manifests, pipeline, task_families
from qrc_vs_qelm.checkpointing import run_chunked
from qrc_vs_qelm.core import readout, tasks as core_tasks
from qrc_vs_qelm.manifests import sha256_file, sha256_frame, write_manifest
from qrc_vs_qelm.matching import (
    assert_bundles_matched,
    assert_features_live,
    assert_matched,
    effective_dimension,
    multiplex_efficacy,
    recommend_T_evolve,
)
from qrc_vs_qelm.models.api import FeatureBundle, QELMConfig, QRCConfig, build_features
from qrc_vs_qelm.seeds import SEED_GRIDS, SeedCell, grid_manifest


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
TIMING_PILOT_PATH = ROOT / "results_freeze" / "timing_pilot.json"
STAGE1_SCORE_STEM = "stage1_scores"
SHRINK_RULE_MESSAGE = (
    "Projected runtime exceeds the Stage-1 budget; apply the preregistered "
    "shrink rule: cut input seeds first, then the secondary product family. "
    "Do not change gates, phase-diagram cells, fairness rules, shot tiers, or "
    "win conditions."
)

SCORE_COLUMNS = [
    "cell_id",
    "input_seed",
    "coupling_seed",
    "shot_seed",
    "model",
    "target",
    "target_family",
    "family_cell",
    "confirmatory",
    "tau",
    "p",
    "capacity_test",
    "nrmse_test",
    "alpha",
    "effective_rank",
    "effective_rank_diag",
    "multiplex_gain",
    "D",
    "n_obs",
    "n_multiplex",
    "selected_hp",
    "gamma",
    "uninformative",
    "capacity_linear_delay",
    "capacity_poly_delay",
    "complete",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dirty_state = {
        "dirty_tree_allowed": bool(args.allow_dirty),
        **manifests.require_clean_tree(bool(args.allow_dirty)),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    config = _load_config(str(args.profile))
    seed_grid_name = str(args.seed_grid or _default_seed_grid(str(args.profile)))
    cells = _seed_cells(seed_grid_name, config)
    runtime_projection = _assert_runtime_budget(config, cells)

    manifest_state: dict[str, Any] = {
        "cells": {},
        "feature_cache": {},
        "errors": [],
    }

    def evaluate_cells(chunk: list[dict[str, Any]]) -> pd.DataFrame:
        frames = [
            _score_seed_cell(
                cell["seed_cell"],
                cell_id=cell["cell_id"],
                config=config,
                out_dir=out_dir,
                manifest_state=manifest_state,
            )
            for cell in chunk
        ]
        if not frames:
            return pd.DataFrame(columns=SCORE_COLUMNS)
        return _order_scores(pd.concat(frames, ignore_index=True))

    try:
        scores = run_chunked(
            cells,
            evaluate_cells,
            out_dir=out_dir,
            stem=STAGE1_SCORE_STEM,
            chunk_size=int(args.checkpoint_every),
            expected_cols=("cell_id", "model", "target", "capacity_test", "complete"),
            dedup_cols=("cell_id", "model", "target"),
            item_id_col="cell_id",
            item_id_fn=lambda item: item["cell_id"],
        )
    except Exception as exc:
        manifest_state["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        _write_stage1_manifest(
            out_dir,
            config=config,
            seed_grid_name=seed_grid_name,
            dirty_state=dirty_state,
            runtime_projection=runtime_projection,
            manifest_state=manifest_state,
            elapsed_s=float(time.perf_counter() - start),
            scores=None,
        )
        raise

    scores = _mark_uninformative(_order_scores(scores), config)
    _write_scores(scores, out_dir / "stage1_scores.csv")
    _refresh_feature_manifest(out_dir, scores, manifest_state)
    gate_payload = _stage1_gate_payload(scores, config)
    manifest_state["gates"] = gate_payload
    _write_stage1_manifest(
        out_dir,
        config=config,
        seed_grid_name=seed_grid_name,
        dirty_state=dirty_state,
        runtime_projection=runtime_projection,
        manifest_state=manifest_state,
        elapsed_s=float(time.perf_counter() - start),
        scores=scores,
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed-grid", choices=sorted(SEED_GRIDS), default=None)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if int(args.checkpoint_every) < 1:
        raise ValueError("--checkpoint-every must be >= 1")
    return args


def _load_config(profile: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{profile}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if str(config.get("profile")) != str(profile):
        raise ValueError(f"{path} profile mismatch: expected {profile!r}")
    return dict(config)


def _default_seed_grid(profile: str) -> str:
    return "stage1_smoke" if profile == "smoke" else "stage1_full"


def _seed_cells(seed_grid_name: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_cells = list(SEED_GRIDS[seed_grid_name])
    stage_shrink = (
        config.get("runtime_budget", {})
        .get("active_shrink", {})
        .get("stage1", {})
    )
    if stage_shrink.get("applied"):
        input_keep = set(int(v) for v in stage_shrink.get("input_seeds_kept", []))
        coupling_keep = set(int(v) for v in stage_shrink.get("coupling_seeds_kept", []))
        if input_keep:
            raw_cells = [cell for cell in raw_cells if int(cell.input_seed) in input_keep]
        if coupling_keep:
            raw_cells = [
                cell for cell in raw_cells if int(cell.coupling_seed) in coupling_keep
            ]

    return [
        {
            "cell_id": _cell_id(cell),
            "seed_cell": cell,
        }
        for cell in raw_cells
    ]


def _cell_id(cell: SeedCell) -> str:
    return (
        f"input{int(cell.input_seed):04d}_"
        f"coupling{int(cell.coupling_seed)}_"
        f"shot{int(cell.shot_seed):04d}"
    )


def _assert_runtime_budget(
    config: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    runtime_cfg = config.get("runtime_budget", {})
    budget_hours = float(runtime_cfg.get("stage_budget_hours", 10.0))
    projection = _project_runtime(config, cells)
    projection["budget_hours"] = budget_hours
    projection["timing_source"] = str(TIMING_PILOT_PATH)
    if projection.get("total_hours", 0.0) > budget_hours:
        raise SystemExit(SHRINK_RULE_MESSAGE)
    return projection


def _project_runtime(
    config: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not TIMING_PILOT_PATH.exists():
        return {"available": False, "total_s": 0.0, "total_hours": 0.0}
    try:
        from scripts.bench_configs import project_runtime
    except Exception:
        return {"available": False, "total_s": 0.0, "total_hours": 0.0}

    timing = json.loads(TIMING_PILOT_PATH.read_text(encoding="utf-8"))
    n_cells = len(cells)
    qrc_grid = int(config["hp_grids"]["qrc"]["cardinality"])
    qelm_grid = int(config["hp_grids"]["qelm"]["cardinality"])
    n_qubits = int(config["primary"]["n_qubits"])
    qrc_class = "qrc_n4" if n_qubits <= 4 else "qrc_n5"
    qelm_class = "qelm_n5"
    projected = project_runtime(
        timing,
        {
            qrc_class: n_cells * qrc_grid,
            qelm_class: n_cells * qelm_grid,
        },
    )
    projected["available"] = True
    projected["n_seed_cells"] = n_cells
    projected["hp_grid_cells"] = {"qrc": qrc_grid, "qelm": qelm_grid}
    return projected


def _score_seed_cell(
    cell: SeedCell,
    *,
    cell_id: str,
    config: Mapping[str, Any],
    out_dir: Path,
    manifest_state: dict[str, Any],
) -> pd.DataFrame:
    input_seed = int(cell.input_seed)
    coupling_seed = int(cell.coupling_seed)
    shot_seed = int(cell.shot_seed)
    rng = np.random.default_rng(input_seed)
    inputs = rng.uniform(0.0, 1.0, size=int(config["T"]))
    observables = _observables_from_config(config)
    targets, target_meta = _build_targets(inputs, input_seed=input_seed, config=config)
    max_delay = _max_required_delay(config, target_meta)
    splits = pipeline.make_splits(
        pipeline.valid_indices(
            int(config["T"]),
            washout=int(config["washout"]),
            max_delay=max_delay,
            targets=targets,
        )
    )
    alphas = tuple(float(alpha) for alpha in config["hp_grids"]["alpha"])
    selection_targets = {
        name: targets[name]
        for name, meta in target_meta.items()
        if meta["target_family"] == "phase_diagram" and bool(meta["confirmatory"])
    }

    qrc_choice = _select_qrc_hp(
        inputs,
        selection_targets,
        splits,
        alphas,
        config=config,
        coupling_seed=coupling_seed,
        observables=observables,
    )
    qelm_choice = _select_qelm_hp(
        inputs,
        selection_targets,
        splits,
        alphas,
        config=config,
        coupling_seed=coupling_seed,
        observables=observables,
    )
    assert_matched(qrc_choice["config"], qelm_choice["config"], observables=observables)

    qrc_bundle = _selected_or_cached_bundle(
        out_dir,
        "qrc",
        cell_id,
        qrc_choice["config"],
        inputs,
        coupling_seed,
        observables,
        fallback_bundle=qrc_choice["bundle"],
    )
    qelm_bundle = _selected_or_cached_bundle(
        out_dir,
        "qelm",
        cell_id,
        qelm_choice["config"],
        inputs,
        coupling_seed,
        observables,
        fallback_bundle=qelm_choice["bundle"],
    )
    assert_bundles_matched(qrc_bundle, qelm_bundle)
    assert_features_live(qrc_bundle)
    assert_features_live(qelm_bundle)

    D = int(qrc_bundle.features.shape[1])
    control_d = int(qelm_choice["config"].d)
    model_matrices: dict[str, tuple[np.ndarray, int, dict[str, Any], int, int]] = {
        "qrc": (
            qrc_bundle.features,
            int(qrc_bundle.first_valid),
            qrc_choice["selected_hp"],
            int(qrc_bundle.n_obs),
            int(qrc_bundle.n_multiplex),
        ),
        "qelm": (
            qelm_bundle.features,
            int(qelm_bundle.first_valid),
            qelm_choice["selected_hp"],
            int(qelm_bundle.n_obs),
            int(qelm_bundle.n_multiplex),
        ),
        "linear_delay": (
            controls.linear_delay_features(inputs, d=control_d, D=D),
            int(qelm_bundle.first_valid),
            {"d": control_d},
            int(qrc_bundle.n_obs),
            int(qrc_bundle.n_multiplex),
        ),
        "poly_delay": (
            controls.poly_delay_features(inputs, d=control_d, D=D),
            int(qelm_bundle.first_valid),
            {"d": control_d, "degree": 2},
            int(qrc_bundle.n_obs),
            int(qrc_bundle.n_multiplex),
        ),
    }

    scored: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for model_name, (features, first_valid, hp, n_obs, n_multiplex) in model_matrices.items():
        scores = pipeline.evaluate_feature_matrix(
            features,
            targets,
            splits,
            alphas=alphas,
            first_valid=first_valid,
        ).set_index("target")
        scored[model_name] = scores
        diag = effective_dimension(features[splits.train])
        mux = _safe_multiplex_gain(features[splits.train], n_multiplex, n_obs)
        diagnostics[model_name] = {
            "effective_dimension": _jsonable(diag),
            "multiplex": _jsonable(mux),
        }
        for target_name, score_row in scores.iterrows():
            meta = target_meta[target_name]
            rows.append(
                {
                    "cell_id": cell_id,
                    "input_seed": input_seed,
                    "coupling_seed": coupling_seed,
                    "shot_seed": shot_seed,
                    "model": model_name,
                    "target": target_name,
                    "target_family": meta["target_family"],
                    "family_cell": meta["family_cell"],
                    "confirmatory": bool(meta["confirmatory"]),
                    "tau": meta.get("tau"),
                    "p": meta.get("p"),
                    "capacity_test": float(score_row["capacity_test"]),
                    "nrmse_test": float(score_row["nrmse_test"]),
                    "alpha": float(score_row["alpha"]),
                    "effective_rank": float(score_row["effective_rank"]),
                    "effective_rank_diag": float(diag["eff_rank"]),
                    "multiplex_gain": float(mux["gain"]),
                    "D": D,
                    "n_obs": n_obs,
                    "n_multiplex": n_multiplex,
                    "selected_hp": json.dumps(_jsonable(hp), sort_keys=True),
                    "gamma": float(config["primary"]["gamma"]),
                    "uninformative": False,
                    "capacity_linear_delay": np.nan,
                    "capacity_poly_delay": np.nan,
                    "complete": 1,
                }
            )

    rows = _attach_control_capacities(rows, scored)
    feature_hashes = _feature_hashes(out_dir, cell_id)
    manifest_state["feature_cache"][cell_id] = feature_hashes
    manifest_state["cells"][cell_id] = {
        "seed_cell": asdict(cell),
        "selected": {
            "qrc": _jsonable(qrc_choice["selected_hp"]),
            "qelm": _jsonable(qelm_choice["selected_hp"]),
        },
        "selection": {
            "qrc": _jsonable(qrc_choice["selection"]),
            "qelm": _jsonable(qelm_choice["selection"]),
        },
        "splits": {
            "n_train": int(splits.train.size),
            "n_val": int(splits.val.size),
            "n_test": int(splits.test.size),
            "first_train": int(splits.train[0]),
            "last_test": int(splits.test[-1]),
        },
        "feature_cache": feature_hashes,
        "diagnostics": diagnostics,
    }
    return _order_scores(pd.DataFrame(rows))


def _observables_from_config(config: Mapping[str, Any]) -> list[readout.Observable]:
    primary = config["primary"]
    obs_cfg = primary.get("observables", {})
    if obs_cfg.get("family", "pauli") != "pauli":
        raise ValueError("only pauli observables are supported")
    observables = readout.pauli_observables(
        int(primary["n_qubits"]),
        axes=tuple(obs_cfg.get("axes", ("x", "y", "z"))),
        max_weight=int(obs_cfg.get("max_weight", 2)),
    )
    expected_dim = obs_cfg.get("dimension")
    if expected_dim is not None and len(observables) != int(expected_dim):
        raise ValueError(
            f"observable dimension mismatch: config={expected_dim}, built={len(observables)}"
        )
    return observables


def _build_targets(
    inputs: np.ndarray,
    *,
    input_seed: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    phase = config["phase_diagram"]
    taus = [int(tau) for tau in phase["taus"]]
    confirmatory_degrees = [int(p) for p in phase["degrees"]]
    robustness_degrees = [
        int(p) for p in phase.get("robustness_degrees", confirmatory_degrees)
    ]

    targets: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, Any]] = {}
    phase_targets = task_families.build_family_targets(
        inputs,
        taus=taus,
        degrees=robustness_degrees,
    )
    for name, values in phase_targets.items():
        tau, degree = _parse_legendre_name(name)
        targets[name] = values
        meta[name] = {
            "target_family": "phase_diagram",
            "family_cell": f"tau{tau}_p{degree}",
            "tau": tau,
            "p": degree,
            "confirmatory": degree in confirmatory_degrees,
        }

    secondary = config.get("secondary_product_family", {})
    if secondary.get("enabled", False):
        product_specs = [
            (int(tau), int(pair[0]), int(pair[1]))
            for tau in secondary.get("taus", [])
            for pair in secondary.get("degree_pairs", [])
        ]
        product_targets = task_families.build_product_targets(inputs, product_specs)
        for name, values in product_targets.items():
            tau, p1, p2 = _parse_product_name(name)
            targets[name] = values
            meta[name] = {
                "target_family": "product",
                "family_cell": name,
                "tau": tau,
                "p": max(p1, p2),
                "confirmatory": False,
            }

    anchors = set(config.get("anchors", {}).get("families", []))
    anchor_taus = [tau for tau in taus if tau > 0] or [1]
    if "stm" in anchors:
        for name, values in task_families.stm_targets(inputs, anchor_taus).items():
            tau = int(name.rsplit("tau", maxsplit=1)[1])
            targets[name] = values
            meta[name] = _anchor_meta(name, tau=tau, p=1)

    narma_all = task_families.narma_targets(inputs)
    for order in (2, 5, 10):
        name = f"narma{order}"
        if name in anchors and name in narma_all:
            targets[name] = narma_all[name]
            meta[name] = _anchor_meta(name, tau=order, p=2)

    if "parity" in anchors:
        bits = (inputs >= 0.5).astype(float)
        for name, values in task_families.parity_targets(bits, anchor_taus).items():
            tau = int(name.rsplit("tau", maxsplit=1)[1])
            targets[name] = values
            meta[name] = _anchor_meta(name, tau=tau, p=max(2, tau))

    if "mackey_glass" in anchors:
        mg_rng = np.random.default_rng(int(input_seed) + 991_027)
        mg = core_tasks.rescale_unit(
            core_tasks.mackey_glass_series(len(inputs), rng=mg_rng)
        )
        targets["mackey_glass"] = mg.astype(np.float64, copy=False)
        meta["mackey_glass"] = _anchor_meta("mackey_glass", tau=17, p=3)

    if "santa_fe_if_cached" in anchors:
        santa_path = ROOT / "data" / "santa_fe.npy"
        if santa_path.exists():
            santa = task_families.load_santa_fe(santa_path)
            if santa is not None and len(santa) >= len(inputs):
                targets["santa_fe"] = np.asarray(santa[: len(inputs)], dtype=np.float64)
                meta["santa_fe"] = _anchor_meta("santa_fe", tau=1, p=3)

    return targets, meta


def _anchor_meta(name: str, *, tau: int, p: int) -> dict[str, Any]:
    return {
        "target_family": "anchor",
        "family_cell": name,
        "tau": int(tau),
        "p": int(p),
        "confirmatory": False,
    }


def _parse_legendre_name(name: str) -> tuple[int, int]:
    left, p_text = name.split("_p", maxsplit=1)
    tau = int(left.rsplit("tau", maxsplit=1)[1])
    return tau, int(p_text)


def _parse_product_name(name: str) -> tuple[int, int, int]:
    # Current product keys are prod_tau{tau}_p{p1}{p2} for configured specs.
    tau_text, p_text = name.removeprefix("prod_tau").split("_p", maxsplit=1)
    if "_tau" in p_text:
        first, rest = p_text.split("_tau", maxsplit=1)
        p1 = int(first)
        _, p2_text = rest.split("_p", maxsplit=1)
        p2 = int(p2_text)
    else:
        p1 = int(p_text[0])
        p2 = int(p_text[1:])
    return int(tau_text), p1, p2


def _max_required_delay(
    config: Mapping[str, Any],
    target_meta: Mapping[str, Mapping[str, Any]],
) -> int:
    target_delay = max(int(meta.get("tau") or 0) for meta in target_meta.values())
    qelm_ds = [int(d) for d in config["hp_grids"]["qelm"]["d"]]
    qelm_first_valid = max(qelm_ds) - 1 if qelm_ds else 0
    return max(target_delay, qelm_first_valid)


def _qrc_base_config(config: Mapping[str, Any]) -> QRCConfig:
    primary = config["primary"]
    return QRCConfig(
        dt=float(primary["dt"]),
        h_in=1.0,
        n_virtual=int(primary["n_virtual"]),
        n_qubits=int(primary["n_qubits"]),
        J_s=float(primary["J_s"]),
        h=float(primary["h"]),
        gamma=float(primary["gamma"]),
        dissipator=str(primary["dissipator"]),
        quantize_levels=primary.get("quantize_levels"),
    )


def _qelm_base_config(config: Mapping[str, Any]) -> QELMConfig:
    primary = config["primary"]
    return QELMConfig(
        d=int(config["hp_grids"]["qelm"]["d"][0]),
        T_evolve=0.1,
        n_snapshots=int(primary["n_snapshots"]),
        n_qubits=int(primary["n_qubits"]),
        J_s=float(primary["J_s"]),
        h=float(primary["h"]),
        gamma=float(primary["gamma"]),
        dissipator=str(primary["dissipator"]),
        quantize_levels=primary.get("quantize_levels"),
    )


def _select_qrc_hp(
    inputs: np.ndarray,
    targets: Mapping[str, np.ndarray],
    splits: pipeline.Splits,
    alphas: Sequence[float],
    *,
    config: Mapping[str, Any],
    coupling_seed: int,
    observables: Sequence[readout.Observable],
) -> dict[str, Any]:
    grid = config["hp_grids"]["qrc"]
    base = _qrc_base_config(config)
    candidates: list[QRCConfig] = [
        replace(base, dt=float(dt), h_in=float(h_in))
        for dt in grid["dt"]
        for h_in in grid["h_in"]
    ]
    _assert_cardinality("qrc", candidates, int(grid["cardinality"]))
    return _select_hp(
        "qrc",
        candidates,
        inputs,
        targets,
        splits,
        alphas,
        coupling_seed=coupling_seed,
        observables=observables,
    )


def _select_qelm_hp(
    inputs: np.ndarray,
    targets: Mapping[str, np.ndarray],
    splits: pipeline.Splits,
    alphas: Sequence[float],
    *,
    config: Mapping[str, Any],
    coupling_seed: int,
    observables: Sequence[readout.Observable],
) -> dict[str, Any]:
    grid = config["hp_grids"]["qelm"]
    base = _qelm_base_config(config)
    candidates: list[QELMConfig] = []
    for d in grid["d"]:
        d_base = replace(base, d=int(d))
        rec = recommend_T_evolve(d_base, coupling_seed)
        for T_evolve in _qelm_T_values(grid["T_evolve"], rec):
            candidates.append(replace(d_base, T_evolve=T_evolve))
    _assert_cardinality("qelm", candidates, int(grid["cardinality"]))
    return _select_hp(
        "qelm",
        candidates,
        inputs,
        targets,
        splits,
        alphas,
        coupling_seed=coupling_seed,
        observables=observables,
    )


def _qelm_T_values(grid: Mapping[str, Any], recommendation: Mapping[str, Any]) -> list[float]:
    mode = grid.get("mode")
    if mode != "recommend_T_evolve_window_multipliers":
        raise ValueError(f"unsupported QELM T_evolve grid mode: {mode!r}")
    timescale = float(recommendation["timescale"])
    t_lo = float(recommendation["t_lo"])
    t_hi = float(recommendation["t_hi"])
    values = []
    for multiplier in grid["multipliers_of_timescale"]:
        T_evolve = float(multiplier) * timescale
        if not (t_lo <= T_evolve <= t_hi):
            raise ValueError(
                f"T_evolve={T_evolve:.6g} outside recommended window "
                f"[{t_lo:.6g}, {t_hi:.6g}]"
            )
        values.append(T_evolve)
    return values


def _assert_cardinality(name: str, candidates: Sequence[Any], expected: int) -> None:
    if len(candidates) != int(expected):
        raise ValueError(
            f"{name} HP grid cardinality mismatch: built {len(candidates)}, "
            f"expected {expected}"
        )


def _select_hp(
    model: str,
    candidates: Sequence[QRCConfig | QELMConfig],
    inputs: np.ndarray,
    targets: Mapping[str, np.ndarray],
    splits: pipeline.Splits,
    alphas: Sequence[float],
    *,
    coupling_seed: int,
    observables: Sequence[readout.Observable],
) -> dict[str, Any]:
    selection_rows: list[dict[str, Any]] = []
    best: tuple[float, int, FeatureBundle, QRCConfig | QELMConfig] | None = None
    for pos, cfg in enumerate(candidates):
        # HP sweeps are not cached; a mid-chunk interruption reruns this cell's sweep.
        bundle = build_features(cfg, inputs, coupling_seed, observables=observables)
        score = _validation_mean_capacity(
            bundle.features,
            targets,
            splits,
            alphas,
            first_valid=int(bundle.first_valid),
        )
        row = {"candidate_index": pos, "validation_capacity_mean": score, **bundle.config}
        selection_rows.append(row)
        if best is None or score > best[0]:
            best = (score, pos, bundle, cfg)
    assert best is not None
    best_score, best_pos, best_bundle, best_cfg = best
    return {
        "config": best_cfg,
        "bundle": best_bundle,
        "selected_hp": {
            "candidate_index": int(best_pos),
            "validation_capacity_mean": float(best_score),
            **best_bundle.config,
        },
        "selection": selection_rows,
        "model": model,
    }


def _validation_mean_capacity(
    features: np.ndarray,
    targets: Mapping[str, np.ndarray],
    splits: pipeline.Splits,
    alphas: Sequence[float],
    *,
    first_valid: int,
) -> float:
    values = np.asarray(features, dtype=np.float64)
    if splits.all.min() < int(first_valid):
        raise ValueError("selection splits include rows before first_valid")
    Xs = pipeline.standardize_train_only(values, splits)
    capacities = []
    for y in targets.values():
        target = np.asarray(y, dtype=np.float64)
        best = -np.inf
        for alpha in alphas:
            pred = pipeline._fit_predict(  # noqa: SLF001 - shared fitted ridge path.
                Xs[splits.train],
                target[splits.train],
                Xs[splits.val],
                float(alpha),
            )
            best = max(best, readout.capacity(target[splits.val], pred))
        capacities.append(best)
    return float(np.mean(capacities))


def _selected_or_cached_bundle(
    out_dir: Path,
    model: str,
    cell_id: str,
    cfg: QRCConfig | QELMConfig,
    inputs: np.ndarray,
    coupling_seed: int,
    observables: Sequence[readout.Observable],
    *,
    fallback_bundle: FeatureBundle | None = None,
) -> FeatureBundle:
    cache_path = _feature_cache_path(out_dir, model, cell_id)
    cached = _load_cached_bundle(cache_path, model=model, cfg=cfg)
    if cached is not None:
        return cached
    bundle = (
        fallback_bundle
        if fallback_bundle is not None
        else build_features(cfg, inputs, coupling_seed, observables=observables)
    )
    _save_bundle(cache_path, bundle)
    return bundle


def _feature_cache_path(out_dir: Path, model: str, cell_id: str) -> Path:
    return out_dir / "features" / f"{model}_{cell_id}.npz"


def _load_cached_bundle(
    path: Path,
    *,
    model: str,
    cfg: QRCConfig | QELMConfig,
) -> FeatureBundle | None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        return None
    observed = sha256_file(path)
    expected = sidecar.read_text(encoding="utf-8").strip()
    if observed != expected:
        return None
    loaded = _load_bundle(path)
    if loaded.model != model:
        return None
    if loaded.config != asdict(cfg):
        return None
    return loaded


def _save_bundle(path: Path, bundle: FeatureBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(bundle.features, dtype=np.float64),
        couplings=np.asarray(bundle.couplings, dtype=np.float64),
        model=np.array(bundle.model),
        config_json=np.array(json.dumps(_jsonable(bundle.config), sort_keys=True)),
        n_obs=np.array(int(bundle.n_obs)),
        n_multiplex=np.array(int(bundle.n_multiplex)),
        first_valid=np.array(int(bundle.first_valid)),
    )
    path.with_suffix(path.suffix + ".sha256").write_text(
        sha256_file(path) + "\n",
        encoding="utf-8",
    )


def _load_bundle(path: Path) -> FeatureBundle:
    with np.load(path, allow_pickle=False) as data:
        return FeatureBundle(
            features=np.asarray(data["features"], dtype=np.float64),
            model=str(data["model"].item()),
            config=json.loads(str(data["config_json"].item())),
            couplings=np.asarray(data["couplings"], dtype=np.float64),
            n_obs=int(data["n_obs"].item()),
            n_multiplex=int(data["n_multiplex"].item()),
            first_valid=int(data["first_valid"].item()),
        )


def _feature_hashes(out_dir: Path, cell_id: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for model in ("qrc", "qelm"):
        path = _feature_cache_path(out_dir, model, cell_id)
        out[model] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return out


def _refresh_feature_manifest(
    out_dir: Path,
    scores: pd.DataFrame,
    manifest_state: dict[str, Any],
) -> None:
    if scores.empty or "cell_id" not in scores.columns:
        return
    for cell_id in scores["cell_id"].astype(str).drop_duplicates():
        paths = [_feature_cache_path(out_dir, model, cell_id) for model in ("qrc", "qelm")]
        if not all(path.exists() for path in paths):
            continue
        hashes = _feature_hashes(out_dir, cell_id)
        manifest_state.setdefault("feature_cache", {})[cell_id] = hashes
        manifest_state.setdefault("cells", {}).setdefault(cell_id, {})[
            "feature_cache"
        ] = hashes


def _safe_multiplex_gain(features: np.ndarray, n_multiplex: int, n_obs: int) -> dict[str, float]:
    if int(n_multiplex) <= 1:
        return {
            "r_full": float(effective_dimension(features)["eff_rank"]),
            "r_single_mean": float(effective_dimension(features)["eff_rank"]),
            "gain": 1.0,
        }
    return multiplex_efficacy(features, int(n_multiplex), int(n_obs))


def _attach_control_capacities(
    rows: list[dict[str, Any]],
    scored: Mapping[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    linear = scored["linear_delay"]
    poly = scored["poly_delay"]
    for row in rows:
        target = row["target"]
        row["capacity_linear_delay"] = float(linear.loc[target, "capacity_test"])
        row["capacity_poly_delay"] = float(poly.loc[target, "capacity_test"])
    return rows


def _mark_uninformative(
    scores: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Mark rows by the frozen family-cell mean-over-seeds rule.

    A confirmatory phase-diagram ``family_cell`` is uninformative iff both QRC
    and QELM mean test capacities across that cell's seed replicates are below
    the configured threshold. This is not a per-seed-observation rule; every row
    for an excluded confirmatory family cell receives the same CSV flag.
    """

    out = scores.copy()
    if "uninformative" not in out.columns:
        out["uninformative"] = False
    out["uninformative"] = False
    threshold = _uninformative_threshold(config)
    if out.empty:
        return out

    confirm = out[
        (out["target_family"].astype(str) == "phase_diagram")
        & (out["confirmatory"].astype(bool))
        & (out["model"].astype(str).isin(("qrc", "qelm")))
    ]
    if confirm.empty:
        return out

    means = (
        confirm.groupby(["family_cell", "model"], sort=False)["capacity_test"]
        .mean()
        .unstack()
    )
    if not {"qrc", "qelm"} <= set(means.columns):
        return out

    uninformative_cells = set(
        means.index[
            means[["qrc", "qelm"]].lt(threshold).all(axis=1).to_numpy(dtype=bool)
        ].astype(str)
    )
    mask = (
        (out["target_family"].astype(str) == "phase_diagram")
        & (out["confirmatory"].astype(bool))
        & (out["family_cell"].astype(str).isin(uninformative_cells))
    )
    out.loc[mask, "uninformative"] = True
    return out


def _stage1_gate_payload(scores: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    wide = _gate_input_frame(scores, config)
    expected = int(config["phase_diagram"]["gated_cells"])
    observed = int(wide["family_cell"].nunique())
    if observed != expected:
        raise ValueError(
            f"Stage-1 FDR family integrity violation: expected {expected} "
            f"confirmatory family cells, observed {observed}"
        )
    n_boot = 200 if config["profile"] == "smoke" else 2000
    threshold = _uninformative_threshold(config)
    full_records = gates.evaluate_stage1_gates(wide, n_boot=n_boot)
    records = _records_with_uninformative_exclusion(
        full_records,
        wide,
        seed=gates.BOOTSTRAP_SEED,
        n_boot=n_boot,
    )
    informative_cells = sorted(
        wide.loc[~wide["uninformative"], "family_cell"].astype(str).unique()
    )
    uninformative_cells = sorted(
        wide.loc[wide["uninformative"], "family_cell"].astype(str).unique()
    )
    excluded_report = _uninformative_cell_capacity_report(wide, threshold)
    return {
        "fdr_family_expected_cells": expected,
        "fdr_family_observed_cells": observed,
        "uninformative_cells": uninformative_cells,
        "uninformative_cell_capacities": excluded_report,
        "informative_cells": informative_cells,
        "g1b_fraction_with_exclusion": _gate_value(records, "G1b"),
        "g1b_fraction_without_exclusion": _gate_value(full_records, "G1b"),
        "g1c_fraction_with_exclusion": _gate_value(records, "G1c"),
        "g1c_fraction_without_exclusion": _gate_value(full_records, "G1c"),
        "records": [_gate_record_dict(record) for record in records],
        "records_full_unfiltered": [
            _gate_record_dict(record) for record in full_records
        ],
        "stage_verdict": gates.stage_verdict(records, stage=1),
        "notes": (
            "BH-FDR was evaluated once on the full confirmatory phase-diagram "
            "family. G1b/G1c fraction records exclude cells where both model "
            "mean test capacities are below the frozen uninformative threshold."
        ),
    }


def _uninformative_threshold(config: Mapping[str, Any]) -> float:
    return float(
        config["phase_diagram"]
        .get("uninformative_exclusion", {})
        .get("both_models_test_capacity_lt", 0.05)
    )


def _gate_value(records: Sequence[gates.GateRecord], gate_id: str) -> Any:
    for record in records:
        if record.gate_id == gate_id:
            return _jsonable(record.value)
    return None


def _uninformative_cell_capacity_report(
    wide: pd.DataFrame,
    threshold: float,
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    means = (
        wide.loc[wide["uninformative"].astype(bool)]
        .groupby("family_cell", sort=True)[["capacity_qrc", "capacity_qelm"]]
        .mean()
    )
    for family_cell, row in means.iterrows():
        report.append(
            {
                "family_cell": str(family_cell),
                "capacity_qrc_mean": float(row["capacity_qrc"]),
                "capacity_qelm_mean": float(row["capacity_qelm"]),
                "threshold": float(threshold),
            }
        )
    return report


def _records_with_uninformative_exclusion(
    full_records: Sequence[gates.GateRecord],
    wide: pd.DataFrame,
    *,
    seed: int,
    n_boot: int,
) -> list[gates.GateRecord]:
    informative = wide.loc[~wide["uninformative"].astype(bool)].copy()
    informative_cells = set(informative["family_cell"].astype(str).unique())
    g1a_records = [record for record in full_records if record.gate_id == "G1a"]
    labels: list[str] = []
    direction_labels: list[str] = []
    for record in g1a_records:
        parsed = _parse_g1a_notes(record.notes)
        family_cell = parsed.get("family_cell")
        label = parsed.get("label", "tie")
        if family_cell not in informative_cells:
            continue
        direction_labels.append(label)
        labels.append(label if record.passed and label in {"QRC", "QELM"} else "tie")

    g1b = gates._evaluate_g1b(  # noqa: SLF001 - keeps G1a FDR on full family.
        labels,
        direction_labels=direction_labels,
    )
    g1b = gates.GateRecord(
        gate_id=g1b.gate_id,
        passed=g1b.passed,
        verdict=g1b.verdict,
        value=g1b.value,
        threshold=g1b.threshold,
        ci_low=g1b.ci_low,
        ci_high=g1b.ci_high,
        q_value=g1b.q_value,
        notes=g1b.notes + "; uninformative_cells_excluded=True",
    )
    g1c = gates._evaluate_g1c(  # noqa: SLF001 - independent of the G1a FDR family.
        informative,
        seed=seed,
        n_boot=n_boot,
    )
    return [*g1a_records, g1b, g1c]


def _parse_g1a_notes(notes: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in str(notes).split():
        if "=" not in token:
            continue
        key, value = token.split("=", maxsplit=1)
        parsed[key] = value
    return parsed


def _gate_record_dict(record: gates.GateRecord) -> dict[str, Any]:
    return {
        "gate_id": record.gate_id,
        "passed": bool(record.passed),
        "verdict": record.verdict,
        "value": _jsonable(record.value),
        "threshold": _jsonable(record.threshold),
        "ci_low": _jsonable(record.ci_low),
        "ci_high": _jsonable(record.ci_high),
        "q_value": _jsonable(record.q_value),
        "notes": record.notes,
    }


def _gate_input_frame(scores: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Return confirmatory family-cell gate input with mean-over-seeds flags.

    ``uninformative`` is recomputed at the ``family_cell`` level: both QRC and
    QELM mean test capacities over all seed replicates must be below the frozen
    threshold. The flag is not inherited from per-row observations.
    """

    confirm = scores[
        (scores["target_family"].astype(str) == "phase_diagram")
        & (scores["confirmatory"].astype(bool))
    ].copy()
    key_cols = [
        "cell_id",
        "input_seed",
        "coupling_seed",
        "shot_seed",
        "target",
        "family_cell",
        "tau",
        "p",
    ]
    pivot = confirm.pivot_table(
        index=key_cols,
        columns="model",
        values="capacity_test",
        aggfunc="first",
    ).reset_index()
    required_models = {"qrc", "qelm", "linear_delay", "poly_delay"}
    missing = sorted(required_models - set(pivot.columns))
    if missing:
        raise ValueError(f"missing models in Stage-1 gate input: {missing}")
    pivot = pivot.rename(
        columns={
            "qrc": "capacity_qrc",
            "qelm": "capacity_qelm",
            "linear_delay": "capacity_linear_delay",
            "poly_delay": "capacity_poly_delay",
        }
    )
    threshold = _uninformative_threshold(config)
    cell_uninformative = (
        pivot.groupby("family_cell", sort=False)[["capacity_qrc", "capacity_qelm"]]
        .mean()
        .lt(threshold)
        .all(axis=1)
    )
    pivot["uninformative"] = pivot["family_cell"].map(cell_uninformative).astype(bool)
    pivot["delta"] = pivot["capacity_qrc"] - pivot["capacity_qelm"]
    return pivot


def _write_stage1_manifest(
    out_dir: Path,
    *,
    config: Mapping[str, Any],
    seed_grid_name: str,
    dirty_state: Mapping[str, Any],
    runtime_projection: Mapping[str, Any],
    manifest_state: Mapping[str, Any],
    elapsed_s: float,
    scores: pd.DataFrame | None,
) -> Path:
    outputs: dict[str, Any] = {
        "scores": str(out_dir / "stage1_scores.csv"),
        "checkpoints": str(out_dir / "checkpoints"),
        "features": str(out_dir / "features"),
    }
    if scores is not None and (out_dir / "stage1_scores.csv").exists():
        outputs["stage1_scores_sha256"] = sha256_file(out_dir / "stage1_scores.csv")
        outputs["stage1_scores_frame_sha256"] = sha256_frame(scores)
        outputs["n_score_rows"] = int(len(scores))

    return write_manifest(
        out_dir,
        "stage1",
        {
            "repo_root": ROOT,
            "stage": 1,
            "profile": config["profile"],
            "config": _jsonable(config),
            "seed_grid": grid_manifest(seed_grid_name),
            "runtime_projection": _jsonable(runtime_projection),
            "runtime": {"elapsed_s": float(elapsed_s)},
            "dirty_state": dict(dirty_state),
            "manifest_state": _jsonable(manifest_state),
            "outputs": outputs,
        },
    )


def _write_scores(scores: pd.DataFrame, path: Path) -> None:
    _order_scores(scores).to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )


def _order_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    out = scores.copy()
    for col in SCORE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out.loc[:, SCORE_COLUMNS].sort_values(
        ["cell_id", "target_family", "target", "model"],
        kind="mergesort",
    ).reset_index(drop=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
