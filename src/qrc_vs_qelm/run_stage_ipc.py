"""IPC window-sweep runner for the QRC ceiling vs windowed TD-QELM."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrc_vs_qelm import ipc, manifests, pipeline, run_stage1, run_stage3
from qrc_vs_qelm.checkpointing import run_chunked
from qrc_vs_qelm.manifests import sha256_file, sha256_frame, write_manifest
from qrc_vs_qelm.matching import (
    MatchingError,
    assert_bundles_matched,
    assert_features_live,
    assert_matched,
)
from qrc_vs_qelm.models.api import FeatureBundle, windowed_qelm_from_qrc
from qrc_vs_qelm.seeds import SEED_GRIDS, SeedCell, grid_manifest


ROOT = Path(__file__).resolve().parents[2]
IPC_SCORE_STEM = "ipc_scores"
SHRINK_RULE_MESSAGE = (
    "Projected runtime exceeds the IPC window-sweep budget; apply the "
    "preregistered shrink rule: shrink the IPC window range first, then cut "
    "seed cells. Do not change the gamma grid, dissipator grid, IPC basis, "
    "fairness rules, or win conditions."
)

IPC_SCORE_COLUMNS = [
    "cell_id",
    "input_seed",
    "coupling_seed",
    "n_qubits",
    "window_k",
    "gamma",
    "dissipator",
    "model",
    "ipc_total",
    "ipc_mc",
    "ipc_nc",
    "n_significant",
    "threshold_saturated",
    "threshold",
    "surrogate_retained_capacity_max",
    "surrogate_retained_capacity_mean",
    "D",
    "delta_ipc",
    "delta_mc",
    "delta_nc",
    "features_dead",
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
    cells = _ipc_seed_cells(seed_grid_name, config)
    window_grid = _window_grid(config)
    basis_caps = _basis_caps(config)
    basis = _ipc_basis(config)
    items = _ipc_items(cells, config, window_grid=window_grid)
    runtime_projection = _assert_runtime_budget(config, items, cells, window_grid)
    manifest_state: dict[str, Any] = {
        "cells": {},
        "feature_cache": {},
        "qrc_ipc_cache": {},
        "errors": [],
    }

    def evaluate_items(chunk: list[dict[str, Any]]) -> pd.DataFrame:
        frames = [
            _score_ipc_item(
                item,
                config=config,
                out_dir=out_dir,
                manifest_state=manifest_state,
                window_grid=window_grid,
                basis=basis,
                basis_caps=basis_caps,
            )
            for item in chunk
        ]
        if not frames:
            return pd.DataFrame(columns=[*IPC_SCORE_COLUMNS, "ipc_item_id"])
        return _order_ipc_scores(pd.concat(frames, ignore_index=True), keep_item_id=True)

    try:
        scores = run_chunked(
            items,
            evaluate_items,
            out_dir=out_dir,
            stem=IPC_SCORE_STEM,
            chunk_size=int(args.checkpoint_every),
            expected_cols=(
                "ipc_item_id",
                "cell_id",
                "model",
                "window_k",
                "gamma",
                "dissipator",
                "ipc_total",
                "delta_ipc",
                "features_dead",
                "complete",
            ),
            dedup_cols=("ipc_item_id", "model"),
            item_id_col="ipc_item_id",
            item_id_fn=lambda item: item["item_id"],
        )
    except Exception as exc:
        manifest_state["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        _write_ipc_manifest(
            out_dir,
            config=config,
            seed_grid_name=seed_grid_name,
            dirty_state=dirty_state,
            runtime_projection=runtime_projection,
            manifest_state=manifest_state,
            elapsed_s=float(time.perf_counter() - start),
            scores=None,
            window_grid=window_grid,
            basis=basis,
            basis_caps=basis_caps,
        )
        raise

    scores = _order_ipc_scores(scores, keep_item_id=False)
    _write_scores(scores, out_dir / "ipc_scores.csv")
    _refresh_feature_manifest(out_dir, scores, manifest_state)
    _write_ipc_manifest(
        out_dir,
        config=config,
        seed_grid_name=seed_grid_name,
        dirty_state=dirty_state,
        runtime_projection=runtime_projection,
        manifest_state=manifest_state,
        elapsed_s=float(time.perf_counter() - start),
        scores=scores,
        window_grid=window_grid,
        basis=basis,
        basis_caps=basis_caps,
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


def _ipc_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    ipc_cfg = config.get("ipc")
    if not isinstance(ipc_cfg, Mapping):
        raise ValueError("config must define an ipc section")
    return ipc_cfg


def _window_grid(config: Mapping[str, Any]) -> list[int]:
    raw = [int(k) for k in _ipc_config(config).get("window_grid", [])]
    if not raw:
        raise ValueError("config.ipc.window_grid must not be empty")
    if any(k < 1 for k in raw):
        raise ValueError("config.ipc.window_grid values must be >= 1")
    if len(set(raw)) != len(raw):
        raise ValueError("config.ipc.window_grid must not contain duplicates")

    shrink = _ipc_shrink(config)
    if shrink.get("applied"):
        keep = {int(k) for k in shrink.get("window_grid_kept", [])}
        if keep:
            raw = [k for k in raw if k in keep]
        if not raw:
            raise ValueError("IPC shrink removed every window_grid value")
    return raw


def _basis_caps(config: Mapping[str, Any]) -> dict[str, int]:
    raw = _ipc_config(config).get("basis", {})
    required = ("max_delay", "max_degree", "max_total_degree", "max_functions")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"config.ipc.basis missing keys: {', '.join(missing)}")
    return {name: int(raw[name]) for name in required}


def _ipc_basis(config: Mapping[str, Any]) -> ipc.Basis:
    caps = _basis_caps(config)
    return ipc.dambre_basis(
        max_delay=caps["max_delay"],
        max_degree=caps["max_degree"],
        max_total_degree=caps["max_total_degree"],
        max_functions=caps["max_functions"],
    )


def _n_surrogates(config: Mapping[str, Any]) -> int:
    return int(_ipc_config(config).get("n_surrogates", ipc.DEFAULT_N_SURROGATES))


def _ipc_shrink(config: Mapping[str, Any]) -> Mapping[str, Any]:
    active = config.get("runtime_budget", {}).get("active_shrink", {})
    if not isinstance(active, Mapping):
        return {}
    for key in ("stage_ipc", "ipc"):
        value = active.get(key, {})
        if isinstance(value, Mapping) and value:
            return value
    return {}


def _ipc_seed_cells(
    seed_grid_name: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_cells = list(SEED_GRIDS[seed_grid_name])
    shrink = _ipc_shrink(config)
    if shrink.get("applied"):
        input_keep = set(int(v) for v in shrink.get("input_seeds_kept", []))
        coupling_keep = set(int(v) for v in shrink.get("coupling_seeds_kept", []))
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


def _ipc_items(
    cells: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    window_grid: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    windows = [int(k) for k in (window_grid if window_grid is not None else _window_grid(config))]
    gamma_grid = [float(gamma) for gamma in config.get("gamma_grid", [])]
    dissipator_grid = [str(dissipator) for dissipator in config.get("dissipator_grid", [])]
    if not gamma_grid:
        raise ValueError("config must define gamma_grid")
    if not dissipator_grid:
        raise ValueError("config must define dissipator_grid")

    items: list[dict[str, Any]] = []
    for window_k in windows:
        for gamma in gamma_grid:
            for dissipator in dissipator_grid:
                for cell in cells:
                    item_id = (
                        f"{cell['cell_id']}_k{int(window_k):02d}_"
                        f"gamma{run_stage3._float_token(gamma)}_diss{dissipator}"
                    )
                    items.append(
                        {
                            **cell,
                            "window_k": int(window_k),
                            "gamma": gamma,
                            "dissipator": dissipator,
                            "item_id": item_id,
                        }
                    )
    return items


def _assert_runtime_budget(
    config: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    window_grid: Sequence[int],
) -> dict[str, Any]:
    runtime_cfg = config.get("runtime_budget", {})
    budget_hours = float(runtime_cfg.get("stage_budget_hours", 10.0))
    projection = _project_runtime(config, items, cells, window_grid)
    projection["budget_hours"] = budget_hours
    projection["timing_source"] = str(run_stage1.TIMING_PILOT_PATH)
    if projection.get("total_hours", 0.0) > budget_hours:
        raise SystemExit(SHRINK_RULE_MESSAGE)
    return projection


def _project_runtime(
    config: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    window_grid: Sequence[int],
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
    n_qubits = int(config["primary"]["n_qubits"])
    qrc_class = "qrc_n4" if n_qubits <= 4 else "qrc_n5"
    qelm_class = "qelm_n5"
    n_gamma = len(config.get("gamma_grid", []))
    n_dissipators = len(config.get("dissipator_grid", []))
    qrc_builds = int(len(cells) * n_gamma * n_dissipators)
    qelm_builds = int(len(items))
    projected = project_runtime(
        timing,
        {
            qrc_class: qrc_builds,
            qelm_class: qelm_builds,
        },
    )
    projected["available"] = True
    projected["n_items"] = int(len(items))
    projected["grid_cells"] = {
        "window_gamma_dissipator_seed_items": int(len(items)),
        "window_grid": [int(k) for k in window_grid],
        "qrc_feature_builds": qrc_builds,
        "qelm_feature_builds": qelm_builds,
    }
    return projected


def _score_ipc_item(
    item: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    out_dir: Path,
    manifest_state: dict[str, Any],
    window_grid: Sequence[int] | None = None,
    basis: Sequence[ipc.BasisSpec] | None = None,
    basis_caps: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    cell: SeedCell = item["seed_cell"]
    input_seed = int(cell.input_seed)
    coupling_seed = int(cell.coupling_seed)
    cell_id = str(item["cell_id"])
    item_id = str(item["item_id"])
    window_k = int(item["window_k"])
    gamma = float(item["gamma"])
    dissipator = str(item["dissipator"])
    windows = [int(k) for k in (window_grid if window_grid is not None else _window_grid(config))]
    max_window_k = max(windows)

    rng = np.random.default_rng(input_seed)
    inputs = rng.uniform(0.0, 1.0, size=int(config["T"]))
    basis_specs = _ipc_basis(config) if basis is None else basis
    caps = _basis_caps(config) if basis_caps is None else dict(basis_caps)
    if int(caps["max_delay"]) < max_window_k - 1:
        raise ValueError("config.ipc.basis.max_delay must cover the largest window first_valid")
    basis_targets = {spec.key: ipc.basis_target(inputs, spec) for spec in basis_specs}
    alphas = tuple(float(alpha) for alpha in config["hp_grids"]["alpha"])
    n_surrogates = _n_surrogates(config)

    structural = run_stage3._fixed_configs(
        config,
        gamma=gamma,
        dissipator=dissipator,
    )
    qrc_cfg = structural["qrc"]
    qelm_cfg = windowed_qelm_from_qrc(qrc_cfg, window_k=window_k)
    observables = run_stage1._observables_from_config(config)
    assert_matched(qrc_cfg, qelm_cfg, observables=observables)

    qrc_cache_id = _qrc_cache_id(cell_id, gamma=gamma, dissipator=dissipator)
    qelm_cache_id = _qelm_cache_id(
        cell_id,
        window_k=window_k,
        gamma=gamma,
        dissipator=dissipator,
    )
    qrc_bundle = run_stage1._selected_or_cached_bundle(
        out_dir,
        "qrc",
        qrc_cache_id,
        qrc_cfg,
        inputs,
        coupling_seed,
        observables,
    )
    qelm_bundle = run_stage1._selected_or_cached_bundle(
        out_dir,
        "qelm",
        qelm_cache_id,
        qelm_cfg,
        inputs,
        coupling_seed,
        observables,
    )
    assert_bundles_matched(qrc_bundle, qelm_bundle)

    liveness = {
        "qrc": _liveness_state(qrc_bundle),
        "qelm": _liveness_state(qelm_bundle),
    }
    splits = _shared_ipc_splits(
        qrc_bundle,
        qelm_bundle,
        targets=basis_targets,
        washout=int(config["washout"]),
        basis_max_delay=int(caps["max_delay"]),
        max_window_k=max_window_k,
    )

    threshold_seed = _stable_seed(
        "ipc-threshold",
        input_seed,
        coupling_seed,
        run_stage3._float_token(gamma),
        dissipator,
    )
    qrc_score_key = _qrc_score_key(
        cell_id,
        gamma=gamma,
        dissipator=dissipator,
        threshold_seed=threshold_seed,
    )
    qrc_score = manifest_state.setdefault("qrc_ipc_cache", {}).get(qrc_score_key)
    if qrc_score is None:
        qrc_score = _score_bundle_ipc(
            qrc_bundle,
            inputs,
            basis_specs,
            splits,
            alphas=alphas,
            n_surrogates=n_surrogates,
            rng_seed=threshold_seed,
            features_dead=bool(liveness["qrc"]["features_dead"]),
        )
        manifest_state["qrc_ipc_cache"][qrc_score_key] = run_stage1._jsonable(qrc_score)

    qelm_score = _score_bundle_ipc(
        qelm_bundle,
        inputs,
        basis_specs,
        splits,
        alphas=alphas,
        n_surrogates=n_surrogates,
        rng_seed=threshold_seed,
        features_dead=bool(liveness["qelm"]["features_dead"]),
    )
    delta = {
        "delta_ipc": float(qrc_score["ipc_total"] - qelm_score["ipc_total"]),
        "delta_mc": float(qrc_score["ipc_mc"] - qelm_score["ipc_mc"]),
        "delta_nc": float(qrc_score["ipc_nc"] - qelm_score["ipc_nc"]),
    }

    D = int(qrc_bundle.features.shape[1])
    common = {
        "cell_id": cell_id,
        "input_seed": input_seed,
        "coupling_seed": coupling_seed,
        "n_qubits": int(qrc_cfg.n_qubits),
        "window_k": window_k,
        "gamma": gamma,
        "dissipator": dissipator,
        "D": D,
        **delta,
        "complete": 1,
        "ipc_item_id": item_id,
    }
    rows = [
        {
            **common,
            "model": "qrc",
            **_score_output_columns(qrc_score),
            "features_dead": bool(liveness["qrc"]["features_dead"]),
        },
        {
            **common,
            "model": "qelm",
            **_score_output_columns(qelm_score),
            "features_dead": bool(liveness["qelm"]["features_dead"]),
        },
    ]

    feature_hashes = {
        "qrc": _feature_hash(out_dir, "qrc", qrc_cache_id),
        "qelm": _feature_hash(out_dir, "qelm", qelm_cache_id),
    }
    manifest_state["feature_cache"][item_id] = feature_hashes
    manifest_state["cells"][item_id] = {
        "seed_cell": {
            "input_seed": input_seed,
            "coupling_seed": coupling_seed,
            "shot_seed": int(cell.shot_seed),
        },
        "window_k": window_k,
        "window_grid": [int(k) for k in windows],
        "gamma": gamma,
        "dissipator": dissipator,
        "basis": _basis_manifest(basis_specs, caps),
        "alphas": [float(alpha) for alpha in alphas],
        "n_surrogates": int(n_surrogates),
        "threshold_seed": int(threshold_seed),
        "fixed_structural_hp": run_stage1._jsonable(run_stage3._fixed_structural_hp(config)),
        "qrc_cache_id": qrc_cache_id,
        "qelm_cache_id": qelm_cache_id,
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
        "splits": _splits_manifest(splits),
        "shared_valid_index_sha256": _index_sha256(splits.all),
        "basis_edge_capacity": {
            "qrc": float(qrc_score["basis_edge_capacity"]),
            "qelm": float(qelm_score["basis_edge_capacity"]),
        },
        "threshold_saturated": {
            "qrc": bool(qrc_score["threshold_saturated"]),
            "qelm": bool(qelm_score["threshold_saturated"]),
        },
        "thresholds_by_degree": {
            "qrc": qrc_score["thresholds_by_degree"],
            "qelm": qelm_score["thresholds_by_degree"],
        },
        "surrogate_retained_capacity_max": {
            "qrc": float(qrc_score["surrogate_retained_capacity_max"]),
            "qelm": float(qelm_score["surrogate_retained_capacity_max"]),
        },
        "surrogate_retained_capacity_mean": {
            "qrc": float(qrc_score["surrogate_retained_capacity_mean"]),
            "qelm": float(qelm_score["surrogate_retained_capacity_mean"]),
        },
        "basis_edge_warning_fraction": float(ipc.DEFAULT_BASIS_EDGE_WARNING_FRACTION),
    }
    return _order_ipc_scores(pd.DataFrame(rows), keep_item_id=True)


def _shared_ipc_splits(
    qrc_bundle: FeatureBundle,
    qelm_bundle: FeatureBundle,
    *,
    targets: Mapping[str, np.ndarray],
    washout: int,
    basis_max_delay: int,
    max_window_k: int,
) -> pipeline.Splits:
    max_first_valid = int(max_window_k) - 1
    if max_first_valid < 0:
        raise ValueError("max_window_k must be >= 1")
    most_restrictive = replace(qelm_bundle, first_valid=max_first_valid)
    idx = pipeline.stage_valid_indices(
        [qrc_bundle, qelm_bundle, most_restrictive],
        targets,
        washout=int(washout),
        max_delay=int(basis_max_delay),
    )
    return pipeline.make_splits(idx)


def _liveness_state(bundle: FeatureBundle) -> dict[str, Any]:
    try:
        assert_features_live(bundle)
    except MatchingError as exc:
        return {"features_dead": True, "liveness_failure": str(exc)}
    return {"features_dead": False, "liveness_failure": None}


def _score_bundle_ipc(
    bundle: FeatureBundle,
    inputs: np.ndarray,
    basis: Sequence[ipc.BasisSpec],
    splits: pipeline.Splits,
    *,
    alphas: Sequence[float],
    n_surrogates: int,
    rng_seed: int,
    features_dead: bool,
) -> dict[str, Any]:
    if features_dead:
        return _dead_ipc_score(basis)
    result = ipc.total_ipc(
        bundle.features,
        inputs,
        basis,
        splits,
        alphas=alphas,
        n_surrogates=int(n_surrogates),
        rng=np.random.default_rng(int(rng_seed)),
    )
    return {
        "ipc_total": float(result["total_ipc"]),
        "ipc_mc": float(result["mc"]),
        "ipc_nc": float(result["nc"]),
        "n_significant": int(result["n_significant"]),
        "threshold_saturated": bool(result["threshold_saturated"]),
        "threshold": float(result["threshold"]),
        "basis_edge_capacity": float(result["basis_edge_capacity"]),
        "n_basis": int(result["n_basis"]),
        "effective_rank": int(result["effective_rank"]),
        "thresholds_by_degree": result["thresholds_by_degree"],
        "surrogate_retained_capacity_max": float(
            result["surrogate_retained_capacity_max"]
        ),
        "surrogate_retained_capacity_mean": float(
            result["surrogate_retained_capacity_mean"]
        ),
        "truncated": bool(result["truncated"]),
    }


def _dead_ipc_score(basis: Sequence[ipc.BasisSpec]) -> dict[str, Any]:
    return {
        "ipc_total": 0.0,
        "ipc_mc": 0.0,
        "ipc_nc": 0.0,
        "n_significant": 0,
        "threshold_saturated": False,
        "threshold": 0.0,
        "basis_edge_capacity": 0.0,
        "n_basis": int(len(basis)),
        "effective_rank": 0,
        "thresholds_by_degree": {},
        "surrogate_retained_capacity_max": 0.0,
        "surrogate_retained_capacity_mean": 0.0,
        "truncated": bool(getattr(basis, "truncated", False)),
    }


def _score_output_columns(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ipc_total": float(score["ipc_total"]),
        "ipc_mc": float(score["ipc_mc"]),
        "ipc_nc": float(score["ipc_nc"]),
        "n_significant": int(score["n_significant"]),
        "threshold_saturated": bool(score["threshold_saturated"]),
        "threshold": float(score["threshold"]),
        "surrogate_retained_capacity_max": float(
            score["surrogate_retained_capacity_max"]
        ),
        "surrogate_retained_capacity_mean": float(
            score["surrogate_retained_capacity_mean"]
        ),
    }


def _qrc_cache_id(cell_id: str, *, gamma: float, dissipator: str) -> str:
    return f"{cell_id}_gamma{run_stage3._float_token(gamma)}_diss{dissipator}"


def _qelm_cache_id(
    cell_id: str,
    *,
    window_k: int,
    gamma: float,
    dissipator: str,
) -> str:
    return (
        f"{cell_id}_k{int(window_k):02d}_"
        f"gamma{run_stage3._float_token(gamma)}_diss{dissipator}"
    )


def _qrc_score_key(
    cell_id: str,
    *,
    gamma: float,
    dissipator: str,
    threshold_seed: int,
) -> str:
    return (
        f"{_qrc_cache_id(cell_id, gamma=gamma, dissipator=dissipator)}_"
        f"threshold{int(threshold_seed)}"
    )


def _feature_hash(out_dir: Path, model: str, cache_id: str) -> dict[str, str]:
    path = run_stage1._feature_cache_path(out_dir, model, cache_id)
    return {"path": str(path), "sha256": sha256_file(path)}


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _index_sha256(idx: np.ndarray) -> str:
    values = np.asarray(idx, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _splits_manifest(splits: pipeline.Splits) -> dict[str, int]:
    return {
        "n_train": int(splits.train.size),
        "n_val": int(splits.val.size),
        "n_test": int(splits.test.size),
        "first_train": int(splits.train[0]),
        "first_val": int(splits.val[0]),
        "first_test": int(splits.test[0]),
        "last_test": int(splits.test[-1]),
    }


def _basis_manifest(basis: Sequence[ipc.BasisSpec], caps: Mapping[str, int]) -> dict[str, Any]:
    return {
        "caps": {key: int(value) for key, value in caps.items()},
        "n_functions": int(len(basis)),
        "truncated": bool(getattr(basis, "truncated", False)),
    }


def _refresh_feature_manifest(
    out_dir: Path,
    scores: pd.DataFrame,
    manifest_state: dict[str, Any],
) -> None:
    if scores.empty:
        return
    keys = scores[["cell_id", "window_k", "gamma", "dissipator"]].drop_duplicates()
    for row in keys.itertuples(index=False):
        qrc_id = _qrc_cache_id(
            str(row.cell_id),
            gamma=float(row.gamma),
            dissipator=str(row.dissipator),
        )
        qelm_id = _qelm_cache_id(
            str(row.cell_id),
            window_k=int(row.window_k),
            gamma=float(row.gamma),
            dissipator=str(row.dissipator),
        )
        qrc_path = run_stage1._feature_cache_path(out_dir, "qrc", qrc_id)
        qelm_path = run_stage1._feature_cache_path(out_dir, "qelm", qelm_id)
        if not qrc_path.exists() or not qelm_path.exists():
            continue
        item_id = (
            f"{row.cell_id}_k{int(row.window_k):02d}_"
            f"gamma{run_stage3._float_token(float(row.gamma))}_diss{row.dissipator}"
        )
        hashes = {
            "qrc": _feature_hash(out_dir, "qrc", qrc_id),
            "qelm": _feature_hash(out_dir, "qelm", qelm_id),
        }
        manifest_state.setdefault("feature_cache", {})[item_id] = hashes
        manifest_state.setdefault("cells", {}).setdefault(item_id, {})[
            "feature_cache"
        ] = hashes


def _write_ipc_manifest(
    out_dir: Path,
    *,
    config: Mapping[str, Any],
    seed_grid_name: str,
    dirty_state: Mapping[str, Any],
    runtime_projection: Mapping[str, Any],
    manifest_state: Mapping[str, Any],
    elapsed_s: float,
    scores: pd.DataFrame | None,
    window_grid: Sequence[int],
    basis: Sequence[ipc.BasisSpec] | None = None,
    basis_caps: Mapping[str, int] | None = None,
) -> Path:
    outputs: dict[str, Any] = {
        "scores": str(out_dir / "ipc_scores.csv"),
        "checkpoints": str(out_dir / "checkpoints"),
        "features": str(out_dir / "features"),
    }
    if scores is not None and (out_dir / "ipc_scores.csv").exists():
        outputs["ipc_scores_sha256"] = sha256_file(out_dir / "ipc_scores.csv")
        outputs["ipc_scores_frame_sha256"] = sha256_frame(scores)
        outputs["n_score_rows"] = int(len(scores))

    basis_specs = _ipc_basis(config) if basis is None else basis
    caps = _basis_caps(config) if basis_caps is None else dict(basis_caps)
    return write_manifest(
        out_dir,
        "ipc",
        {
            "repo_root": ROOT,
            "stage": "ipc",
            "profile": config["profile"],
            "config": run_stage1._jsonable(config),
            "seed_grid": grid_manifest(seed_grid_name),
            "window_grid": [int(k) for k in window_grid],
            "gamma_grid": [float(gamma) for gamma in config.get("gamma_grid", [])],
            "dissipator_grid": [str(d) for d in config.get("dissipator_grid", [])],
            "basis": _basis_manifest(basis_specs, caps),
            "n_surrogates": _n_surrogates(config),
            "alphas": [float(alpha) for alpha in config["hp_grids"]["alpha"]],
            "runtime_projection": run_stage1._jsonable(runtime_projection),
            "runtime": {"elapsed_s": float(elapsed_s)},
            "dirty_state": dict(dirty_state),
            "manifest_state": run_stage1._jsonable(manifest_state),
            "outputs": outputs,
        },
    )


def _write_scores(scores: pd.DataFrame, path: Path) -> None:
    _order_ipc_scores(scores, keep_item_id=False).to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )


def _order_ipc_scores(
    scores: pd.DataFrame,
    *,
    keep_item_id: bool,
) -> pd.DataFrame:
    columns = [*IPC_SCORE_COLUMNS]
    if keep_item_id:
        columns.append("ipc_item_id")
    if scores.empty:
        return pd.DataFrame(columns=columns)
    out = scores.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    ordered = out.loc[:, columns].sort_values(
        ["window_k", "gamma", "dissipator", "cell_id", "model"],
        kind="mergesort",
    )
    return ordered.reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
