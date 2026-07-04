"""Stage-3 runner for the pre-registered RQ3 dissipation sweep."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrc_vs_qelm import manifests, pipeline, run_stage1
from qrc_vs_qelm.checkpointing import run_chunked
from qrc_vs_qelm.manifests import sha256_file, sha256_frame, write_manifest
from qrc_vs_qelm.matching import (
    assert_bundles_matched,
    assert_features_live,
    assert_matched,
    MatchingError,
    recommend_T_evolve,
)
from qrc_vs_qelm.seeds import SEED_GRIDS, SeedCell, grid_manifest


ROOT = Path(__file__).resolve().parents[2]
STAGE3_SCORE_STEM = "stage3_scores"
FIXED_QRC_DT = 0.5
FIXED_QRC_H_IN = 0.5
REQUESTED_QELM_D = 5
STAGE3_QELM_REFERENCE_GAMMA = 0.3
SHRINK_RULE_MESSAGE = (
    "Projected runtime exceeds the Stage-3 budget; apply the preregistered "
    "shrink rule: cut input seeds first, then the secondary product family. "
    "Do not change gates, phase-diagram cells, fairness rules, shot tiers, or "
    "win conditions."
)

STAGE3_SCORE_COLUMNS = [
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
    "features_dead",
    "gamma",
    "dissipator",
    "MC",
    "NC",
    "delta_capacity_qrc_minus_qelm",
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
    config = run_stage1._load_config(str(args.profile))
    seed_grid_name = str(config.get("seed_grids", {}).get("stage3", "stage3_gamma"))
    cells = _stage3_seed_cells(seed_grid_name, config)
    items = _stage3_items(cells, config)
    runtime_projection = _assert_runtime_budget(config, items)
    manifest_state: dict[str, Any] = {
        "cells": {},
        "feature_cache": {},
        "fixed_structural_hp": _fixed_structural_hp(config),
        "errors": [],
    }

    def evaluate_items(chunk: list[dict[str, Any]]) -> pd.DataFrame:
        frames = [
            _score_gamma_item(
                item,
                config=config,
                out_dir=out_dir,
                manifest_state=manifest_state,
            )
            for item in chunk
        ]
        if not frames:
            return pd.DataFrame(columns=[*STAGE3_SCORE_COLUMNS, "stage3_item_id"])
        return _order_stage3_scores(pd.concat(frames, ignore_index=True), keep_item_id=True)

    try:
        scores = run_chunked(
            items,
            evaluate_items,
            out_dir=out_dir,
            stem=STAGE3_SCORE_STEM,
            chunk_size=int(args.checkpoint_every),
            expected_cols=(
                "stage3_item_id",
                "cell_id",
                "model",
                "target",
                "gamma",
                "dissipator",
                "capacity_test",
                "features_dead",
                "MC",
                "NC",
                "complete",
            ),
            dedup_cols=("stage3_item_id", "model", "target"),
            item_id_col="stage3_item_id",
            item_id_fn=lambda item: item["item_id"],
        )
    except Exception as exc:
        manifest_state["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        _write_stage3_manifest(
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

    scores = _attach_mc_nc_and_delta(_order_stage3_scores(scores, keep_item_id=False))
    scores = _order_stage3_scores(scores, keep_item_id=False)
    _write_scores(scores, out_dir / "stage3_scores.csv")
    _refresh_feature_manifest(out_dir, scores, manifest_state)
    _write_stage3_manifest(
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
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if int(args.checkpoint_every) < 1:
        raise ValueError("--checkpoint-every must be >= 1")
    return args


def _stage3_seed_cells(
    seed_grid_name: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_cells = list(SEED_GRIDS[seed_grid_name])
    stage_shrink = (
        config.get("runtime_budget", {})
        .get("active_shrink", {})
        .get("stage3", {})
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
            "cell_id": run_stage1._cell_id(cell),
            "seed_cell": cell,
        }
        for cell in raw_cells
    ]


def _stage3_items(
    cells: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gamma_grid = [float(gamma) for gamma in config.get("gamma_grid", [])]
    dissipator_grid = [str(dissipator) for dissipator in config.get("dissipator_grid", [])]
    if not gamma_grid:
        raise ValueError("config must define gamma_grid")
    if not dissipator_grid:
        raise ValueError("config must define dissipator_grid")

    items: list[dict[str, Any]] = []
    for gamma in gamma_grid:
        for dissipator in dissipator_grid:
            for cell in cells:
                item_id = (
                    f"{cell['cell_id']}_gamma{_float_token(gamma)}_"
                    f"diss{dissipator}"
                )
                items.append(
                    {
                        **cell,
                        "gamma": gamma,
                        "dissipator": dissipator,
                        "item_id": item_id,
                    }
                )
    return items


def _float_token(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def _assert_runtime_budget(
    config: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    runtime_cfg = config.get("runtime_budget", {})
    budget_hours = float(runtime_cfg.get("stage_budget_hours", 10.0))
    projection = _project_runtime(config, items)
    projection["budget_hours"] = budget_hours
    projection["timing_source"] = str(run_stage1.TIMING_PILOT_PATH)
    if projection.get("total_hours", 0.0) > budget_hours:
        raise SystemExit(SHRINK_RULE_MESSAGE)
    return projection


def _project_runtime(
    config: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not run_stage1.TIMING_PILOT_PATH.exists():
        return {
            "available": False,
            "n_items": int(len(items)),
            "total_s": 0.0,
            "total_hours": 0.0,
        }
    try:
        from scripts.bench_configs import project_runtime
    except Exception:
        return {
            "available": False,
            "n_items": int(len(items)),
            "total_s": 0.0,
            "total_hours": 0.0,
        }

    timing = json.loads(run_stage1.TIMING_PILOT_PATH.read_text(encoding="utf-8"))
    n_items = int(len(items))
    n_qubits = int(config["primary"]["n_qubits"])
    qrc_class = "qrc_n4" if n_qubits <= 4 else "qrc_n5"
    qelm_class = "qelm_n5"
    projected = project_runtime(
        timing,
        {
            qrc_class: n_items,
            qelm_class: n_items,
        },
    )
    projected["available"] = True
    projected["n_items"] = n_items
    projected["grid_cells"] = {
        "gamma_dissipator_seed_items": n_items,
        "qrc_feature_builds_per_item": 1,
        "qelm_feature_builds_per_item": 1,
    }
    return projected


def _fixed_structural_hp(config: Mapping[str, Any]) -> dict[str, Any]:
    n_qubits = int(config["primary"]["n_qubits"])
    effective_d = min(REQUESTED_QELM_D, n_qubits)
    return {
        "qrc": {"dt": FIXED_QRC_DT, "h_in": FIXED_QRC_H_IN},
        "qelm": {
            "requested_d": REQUESTED_QELM_D,
            "d": effective_d,
            "d_smoke_deviation": effective_d != REQUESTED_QELM_D,
            "T_evolve": (
                "fixed recommend_T_evolve.t_recommended at gamma=0.3 "
                "for each coupling seed and dissipator"
            ),
        },
    }


def _score_gamma_item(
    item: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    out_dir: Path,
    manifest_state: dict[str, Any],
) -> pd.DataFrame:
    cell: SeedCell = item["seed_cell"]
    input_seed = int(cell.input_seed)
    coupling_seed = int(cell.coupling_seed)
    shot_seed = int(cell.shot_seed)
    cell_id = str(item["cell_id"])
    item_id = str(item["item_id"])
    gamma = float(item["gamma"])
    dissipator = str(item["dissipator"])

    rng = np.random.default_rng(input_seed)
    inputs = rng.uniform(0.0, 1.0, size=int(config["T"]))
    targets_all, target_meta_all = run_stage1._build_targets(
        inputs,
        input_seed=input_seed,
        config=config,
    )
    targets = {
        name: values
        for name, values in targets_all.items()
        if target_meta_all[name]["target_family"] == "phase_diagram"
    }
    target_meta = {name: target_meta_all[name] for name in targets}
    structural = _fixed_configs(config, gamma=gamma, dissipator=dissipator)
    qrc_cfg = structural["qrc"]
    qelm_seed_cfg = structural["qelm_seed"]
    recommendation = _stage3_t_evolve_recommendation(qelm_seed_cfg, coupling_seed)
    qelm_cfg = replace(qelm_seed_cfg, T_evolve=float(recommendation["t_stage3"]))

    observables = run_stage1._observables_from_config(config)
    assert_matched(qrc_cfg, qelm_cfg, observables=observables)
    max_delay = max(
        run_stage1._max_required_delay(config, target_meta),
        int(qelm_cfg.d) - 1,
    )
    splits = pipeline.make_splits(
        pipeline.valid_indices(
            int(config["T"]),
            washout=int(config["washout"]),
            max_delay=max_delay,
            targets=targets,
        )
    )
    alphas = tuple(float(alpha) for alpha in config["hp_grids"]["alpha"])

    qrc_bundle = run_stage1._selected_or_cached_bundle(
        out_dir,
        "qrc",
        item_id,
        qrc_cfg,
        inputs,
        coupling_seed,
        observables,
    )
    qelm_bundle = run_stage1._selected_or_cached_bundle(
        out_dir,
        "qelm",
        item_id,
        qelm_cfg,
        inputs,
        coupling_seed,
        observables,
    )
    assert_bundles_matched(qrc_bundle, qelm_bundle)
    liveness: dict[str, dict[str, Any]] = {}
    for bundle in (qrc_bundle, qelm_bundle):
        try:
            assert_features_live(bundle)
        except MatchingError as exc:
            liveness[bundle.model] = {
                "features_dead": True,
                "liveness_failure": str(exc),
            }
        else:
            liveness[bundle.model] = {
                "features_dead": False,
                "liveness_failure": None,
            }

    D = int(qrc_bundle.features.shape[1])
    model_matrices: dict[str, tuple[np.ndarray, int, dict[str, Any], int, int]] = {
        "qrc": (
            qrc_bundle.features,
            int(qrc_bundle.first_valid),
            {
                **qrc_bundle.config,
                "fixed_stage3": True,
            },
            int(qrc_bundle.n_obs),
            int(qrc_bundle.n_multiplex),
        ),
        "qelm": (
            qelm_bundle.features,
            int(qelm_bundle.first_valid),
            {
                **qelm_bundle.config,
                "fixed_stage3": True,
                "recommend_T_evolve": run_stage1._jsonable(recommendation),
                "requested_d": REQUESTED_QELM_D,
            },
            int(qelm_bundle.n_obs),
            int(qelm_bundle.n_multiplex),
        ),
    }

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for model_name, (features, first_valid, hp, n_obs, n_multiplex) in model_matrices.items():
        features_dead = bool(liveness[model_name]["features_dead"])
        diag, mux = _feature_diagnostics(features, splits, n_multiplex, n_obs)
        diagnostics[model_name] = {
            "effective_dimension": run_stage1._jsonable(diag),
            "multiplex": run_stage1._jsonable(mux),
            "features_dead": features_dead,
            "liveness_failure": liveness[model_name]["liveness_failure"],
        }

        if features_dead:
            indexed_scores = _dead_feature_scores(targets)
        else:
            indexed_scores = pipeline.evaluate_feature_matrix(
                features,
                targets,
                splits,
                alphas=alphas,
                first_valid=first_valid,
            ).set_index("target")

        for target_name, score_row in indexed_scores.iterrows():
            meta = target_meta[target_name]
            rows.append(
                {
                    "stage3_item_id": item_id,
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
                    "n_obs": int(n_obs),
                    "n_multiplex": int(n_multiplex),
                    "selected_hp": json.dumps(run_stage1._jsonable(hp), sort_keys=True),
                    "features_dead": features_dead,
                    "gamma": gamma,
                    "dissipator": dissipator,
                    "MC": np.nan,
                    "NC": np.nan,
                    "delta_capacity_qrc_minus_qelm": np.nan,
                    "uninformative": False,
                    "capacity_linear_delay": np.nan,
                    "capacity_poly_delay": np.nan,
                    "complete": 1,
                }
            )

    frame = _attach_mc_nc_and_delta(pd.DataFrame(rows))
    feature_hashes = _feature_hashes(out_dir, item_id)
    manifest_state["feature_cache"][item_id] = feature_hashes
    manifest_state["cells"][item_id] = {
        "seed_cell": {
            "input_seed": input_seed,
            "coupling_seed": coupling_seed,
            "shot_seed": shot_seed,
        },
        "gamma": gamma,
        "dissipator": dissipator,
        "qelm_recommend_T_evolve": run_stage1._jsonable(recommendation),
        "fixed_structural_hp": run_stage1._jsonable(_fixed_structural_hp(config)),
        "splits": {
            "n_train": int(splits.train.size),
            "n_val": int(splits.val.size),
            "n_test": int(splits.test.size),
            "first_train": int(splits.train[0]),
            "last_test": int(splits.test[-1]),
        },
        "feature_cache": feature_hashes,
        "features_dead": {
            model: bool(state["features_dead"])
            for model, state in liveness.items()
        },
        "liveness_failures": {
            model: state["liveness_failure"]
            for model, state in liveness.items()
            if state["liveness_failure"] is not None
        },
        "diagnostics": diagnostics,
    }
    return _order_stage3_scores(frame, keep_item_id=True)


def _fixed_configs(
    config: Mapping[str, Any],
    *,
    gamma: float,
    dissipator: str,
) -> dict[str, Any]:
    primary = config["primary"]
    n_qubits = int(primary["n_qubits"])
    qelm_d = min(REQUESTED_QELM_D, n_qubits)
    qrc_cfg = run_stage1.QRCConfig(
        dt=FIXED_QRC_DT,
        h_in=FIXED_QRC_H_IN,
        n_virtual=int(primary["n_virtual"]),
        n_qubits=n_qubits,
        J_s=float(primary["J_s"]),
        h=float(primary["h"]),
        gamma=float(gamma),
        dissipator=str(dissipator),
        quantize_levels=primary.get("quantize_levels"),
    )
    qelm_seed_cfg = run_stage1.QELMConfig(
        d=qelm_d,
        T_evolve=0.0,
        n_snapshots=int(primary["n_snapshots"]),
        n_qubits=n_qubits,
        J_s=float(primary["J_s"]),
        h=float(primary["h"]),
        gamma=float(gamma),
        dissipator=str(dissipator),
        quantize_levels=primary.get("quantize_levels"),
    )
    return {"qrc": qrc_cfg, "qelm_seed": qelm_seed_cfg}


def _stage3_t_evolve_recommendation(
    qelm_seed_cfg: Any,
    coupling_seed: int,
) -> dict[str, Any]:
    # Stage 3 holds QELM's per-window horizon fixed at the RQ1 gamma=0.3
    # operating point, matching QRC's fixed dt=0.5-per-symbol design: gamma
    # then changes only the dissipator strength, not a co-varying horizon.
    reference_cfg = replace(qelm_seed_cfg, gamma=STAGE3_QELM_REFERENCE_GAMMA)
    reference = recommend_T_evolve(reference_cfg, coupling_seed)
    t_stage3 = float(reference["t_recommended"])
    return {
        **reference,
        "t_stage3": float(t_stage3),
        "reference_gamma": float(STAGE3_QELM_REFERENCE_GAMMA),
        "reference_dissipator": str(reference_cfg.dissipator),
        "selection_rule": "fixed_gamma_0.3_t_recommended",
    }


def _feature_diagnostics(
    features: np.ndarray,
    splits: pipeline.Splits,
    n_multiplex: int,
    n_obs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = np.asarray(features, dtype=np.float64)[splits.train]
    try:
        diag = run_stage1.effective_dimension(train)
    except Exception as exc:
        diag = {
            "eff_rank": 0.0,
            "eff_rank_frac": 0.0,
            "n_dead_cols": int(features.shape[1]) if features.ndim == 2 else 0,
            "min_col_var": 0.0,
            "error": str(exc),
        }
    try:
        mux = run_stage1._safe_multiplex_gain(train, n_multiplex, n_obs)
    except Exception as exc:
        mux = {"gain": 0.0, "error": str(exc)}
    return diag, mux


def _dead_feature_scores(targets: Mapping[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": target_name,
                "capacity_test": 0.0,
                "nrmse_test": np.inf,
                "alpha": np.nan,
                "effective_rank": 0.0,
            }
            for target_name in targets
        ]
    ).set_index("target")


def _attach_mc_nc_and_delta(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        out = scores.copy()
        for col in ("MC", "NC", "delta_capacity_qrc_minus_qelm"):
            if col not in out.columns:
                out[col] = np.nan
        return out

    out = scores.copy()
    group_cols = _available_group_cols(
        out,
        ["gamma", "dissipator", "cell_id", "input_seed", "coupling_seed", "shot_seed", "model"],
    )
    phase = out[out["target_family"].astype(str) == "phase_diagram"].copy()
    phase["p_numeric"] = pd.to_numeric(phase["p"], errors="coerce")
    summaries = []
    for key, group in phase.groupby(group_cols, sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple, strict=True))
        p = group["p_numeric"]
        row["MC"] = float(group.loc[p == 1, "capacity_test"].sum())
        row["NC"] = float(group.loc[p >= 2, "capacity_test"].sum())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    out = out.drop(columns=[col for col in ("MC", "NC") if col in out.columns])
    if not summary.empty:
        out = out.merge(summary, on=group_cols, how="left")
    else:
        out["MC"] = np.nan
        out["NC"] = np.nan

    delta_key = _available_group_cols(
        out,
        [
            "gamma",
            "dissipator",
            "cell_id",
            "input_seed",
            "coupling_seed",
            "shot_seed",
            "target",
            "family_cell",
            "tau",
            "p",
        ],
    )
    pivot = phase.pivot_table(
        index=delta_key,
        columns="model",
        values="capacity_test",
        aggfunc="first",
    ).reset_index()
    if {"qrc", "qelm"} <= set(pivot.columns):
        pivot["delta_capacity_qrc_minus_qelm"] = pivot["qrc"] - pivot["qelm"]
        delta = pivot[[*delta_key, "delta_capacity_qrc_minus_qelm"]]
        out = out.drop(
            columns=["delta_capacity_qrc_minus_qelm"],
            errors="ignore",
        ).merge(delta, on=delta_key, how="left")
    else:
        out["delta_capacity_qrc_minus_qelm"] = np.nan
    return out


def _available_group_cols(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [col for col in columns if col in frame.columns]


def _feature_hashes(out_dir: Path, item_id: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for model in ("qrc", "qelm"):
        path = run_stage1._feature_cache_path(out_dir, model, item_id)
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
    if scores.empty or "gamma" not in scores.columns:
        return
    keys = scores[["cell_id", "gamma", "dissipator"]].drop_duplicates()
    for row in keys.itertuples(index=False):
        item_id = f"{row.cell_id}_gamma{_float_token(float(row.gamma))}_diss{row.dissipator}"
        paths = [run_stage1._feature_cache_path(out_dir, model, item_id) for model in ("qrc", "qelm")]
        if not all(path.exists() for path in paths):
            continue
        hashes = _feature_hashes(out_dir, item_id)
        manifest_state.setdefault("feature_cache", {})[item_id] = hashes
        manifest_state.setdefault("cells", {}).setdefault(item_id, {})[
            "feature_cache"
        ] = hashes


def _write_stage3_manifest(
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
        "scores": str(out_dir / "stage3_scores.csv"),
        "checkpoints": str(out_dir / "checkpoints"),
        "features": str(out_dir / "features"),
    }
    if scores is not None and (out_dir / "stage3_scores.csv").exists():
        outputs["stage3_scores_sha256"] = sha256_file(out_dir / "stage3_scores.csv")
        outputs["stage3_scores_frame_sha256"] = sha256_frame(scores)
        outputs["n_score_rows"] = int(len(scores))

    return write_manifest(
        out_dir,
        "stage3",
        {
            "repo_root": ROOT,
            "stage": 3,
            "profile": config["profile"],
            "config": run_stage1._jsonable(config),
            "seed_grid": grid_manifest(seed_grid_name),
            "gamma_grid": [float(gamma) for gamma in config.get("gamma_grid", [])],
            "dissipator_grid": [str(d) for d in config.get("dissipator_grid", [])],
            "runtime_projection": run_stage1._jsonable(runtime_projection),
            "runtime": {"elapsed_s": float(elapsed_s)},
            "dirty_state": dict(dirty_state),
            "manifest_state": run_stage1._jsonable(manifest_state),
            "outputs": outputs,
        },
    )


def _write_scores(scores: pd.DataFrame, path: Path) -> None:
    _order_stage3_scores(scores, keep_item_id=False).to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )


def _order_stage3_scores(
    scores: pd.DataFrame,
    *,
    keep_item_id: bool,
) -> pd.DataFrame:
    columns = [*STAGE3_SCORE_COLUMNS]
    if keep_item_id:
        columns.append("stage3_item_id")
    if scores.empty:
        return pd.DataFrame(columns=columns)
    out = scores.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    ordered = out.loc[:, columns].sort_values(
        ["gamma", "dissipator", "cell_id", "target_family", "target", "model"],
        kind="mergesort",
    )
    return ordered.reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
