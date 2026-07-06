#!/usr/bin/env python
"""IPC design pilot for T reliability, signed Delta(k,gamma), and gate freeze.

Pilot outputs are design diagnostics only. They must not be imported into the
confirmatory analysis.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrc_vs_qelm import ipc, run_stage1, run_stage3, run_stage_ipc, stats
from qrc_vs_qelm.core import diagnostics, reservoirs
from qrc_vs_qelm.gates import BOOTSTRAP_SEED
from qrc_vs_qelm.matching import (
    MatchingError,
    assert_bundles_matched,
    assert_features_live,
    assert_matched,
)
from qrc_vs_qelm.models.api import (
    FeatureBundle,
    QRCConfig,
    _dissipator_jumps,
    windowed_qelm_from_qrc,
)
from qrc_vs_qelm.seeds import SEED_GRIDS, SeedCell, grid_manifest


DEFAULT_OUT = ROOT / "results_ipc_pilot"
PILOT_SEED_GRID = "pilot"
TUNED_GAMMA = 0.3
PILOT_DISSIPATOR = "local_loss"
FULL_T_VALUES = (500, 1000, 2000, 4000, 8000)
FULL_RELIABILITY_WINDOWS = (1, 4, 8)
FULL_SURFACE_WINDOWS = (1, 2, 3, 5, 8, 12)
FULL_GAMMAS = (0.02, 0.1, 0.3, 0.5)
SMALL_EDGE_CAPACITY = ipc.DEFAULT_BASIS_EDGE_WARNING_FRACTION


@dataclass(frozen=True)
class IpcPilotProfile:
    """Runtime/grid controls for the pilot script."""

    t_values: tuple[int, ...]
    reliability_windows: tuple[int, ...]
    surface_windows: tuple[int, ...]
    gamma_grid: tuple[float, ...]
    n_boot: int
    n_surrogates: int | None
    max_seed_cells: int | None
    basis_caps: dict[str, int] | None
    n_gap_couplings: int
    stabilization_tol: float = 0.05
    rank_clamp_fraction_max: float = 0.98
    basis_edge_max: float = SMALL_EDGE_CAPACITY


PROFILES: dict[str, IpcPilotProfile] = {
    "smoke": IpcPilotProfile(
        t_values=(72, 96),
        reliability_windows=(1, 3),
        surface_windows=(1, 3),
        gamma_grid=(0.02, 0.3),
        n_boot=64,
        n_surrogates=2,
        max_seed_cells=1,
        basis_caps={
            "max_delay": 4,
            "max_degree": 2,
            "max_total_degree": 2,
            "max_functions": 16,
        },
        n_gap_couplings=1,
    ),
    "full": IpcPilotProfile(
        t_values=FULL_T_VALUES,
        reliability_windows=FULL_RELIABILITY_WINDOWS,
        surface_windows=FULL_SURFACE_WINDOWS,
        gamma_grid=FULL_GAMMAS,
        n_boot=2000,
        n_surrogates=None,
        max_seed_cells=None,
        basis_caps=None,
        n_gap_couplings=1,
    ),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return "inf" if value > 0 else "-inf"
    return value


def _parse_float_csv(value: str) -> tuple[float, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    try:
        parsed = tuple(float(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not np.isfinite(parsed).all():
        raise argparse.ArgumentTypeError("all values must be finite")
    return parsed


def _parse_int_csv(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated int")
    try:
        parsed = tuple(int(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be >= 1")
    return parsed


def _profile_with_overrides(
    profile: IpcPilotProfile,
    *,
    t_values: tuple[int, ...] | None,
    reliability_windows: tuple[int, ...] | None,
    surface_windows: tuple[int, ...] | None,
    gammas: tuple[float, ...] | None,
) -> IpcPilotProfile:
    return IpcPilotProfile(
        t_values=tuple(int(t) for t in (t_values or profile.t_values)),
        reliability_windows=tuple(
            int(k) for k in (reliability_windows or profile.reliability_windows)
        ),
        surface_windows=tuple(int(k) for k in (surface_windows or profile.surface_windows)),
        gamma_grid=tuple(float(gamma) for gamma in (gammas or profile.gamma_grid)),
        n_boot=int(profile.n_boot),
        n_surrogates=profile.n_surrogates,
        max_seed_cells=profile.max_seed_cells,
        basis_caps=copy.deepcopy(profile.basis_caps),
        n_gap_couplings=int(profile.n_gap_couplings),
        stabilization_tol=float(profile.stabilization_tol),
        rank_clamp_fraction_max=float(profile.rank_clamp_fraction_max),
        basis_edge_max=float(profile.basis_edge_max),
    )


def _pilot_cells(max_seed_cells: int | None) -> list[dict[str, Any]]:
    cells = list(SEED_GRIDS[PILOT_SEED_GRID])
    if max_seed_cells is not None:
        cells = cells[: int(max_seed_cells)]
    return [{"cell_id": run_stage1._cell_id(cell), "seed_cell": cell} for cell in cells]


def _washout_for_T(config: Mapping[str, Any], T: int) -> int:
    base_T = max(1, int(config["T"]))
    base_washout = int(config["washout"])
    return int(round(float(T) * float(base_washout) / float(base_T)))


def _config_for_T(
    base_config: Mapping[str, Any],
    *,
    T: int,
    washout: int,
    profile: IpcPilotProfile,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    config["T"] = int(T)
    config["washout"] = int(washout)
    config["gamma_grid"] = [float(gamma) for gamma in profile.gamma_grid]
    config["dissipator_grid"] = [PILOT_DISSIPATOR]
    config.setdefault("ipc", {})
    if profile.basis_caps is not None:
        config["ipc"]["basis"] = copy.deepcopy(profile.basis_caps)
    if profile.n_surrogates is not None:
        config["ipc"]["n_surrogates"] = int(profile.n_surrogates)
    return config


def _basis_from_config(config: Mapping[str, Any]) -> tuple[ipc.Basis, dict[str, int]]:
    caps = run_stage_ipc._basis_caps(config)
    basis = ipc.dambre_basis(
        max_delay=caps["max_delay"],
        max_degree=caps["max_degree"],
        max_total_degree=caps["max_total_degree"],
        max_functions=caps["max_functions"],
    )
    return basis, caps


def _liveness_state(bundle: FeatureBundle) -> dict[str, Any]:
    try:
        assert_features_live(bundle)
    except MatchingError as exc:
        return {"features_dead": True, "liveness_failure": str(exc)}
    return {"features_dead": False, "liveness_failure": None}


def _dead_ipc_score(basis: Sequence[ipc.BasisSpec]) -> dict[str, Any]:
    return {
        "ipc_total": 0.0,
        "ipc_mc": 0.0,
        "ipc_nc": 0.0,
        "raw_total_ipc": 0.0,
        "effective_rank": 0,
        "rank_clamp_fraction": float("inf"),
        "n_significant": 0,
        "threshold_saturated": False,
        "threshold": 0.0,
        "basis_edge_capacity": 0.0,
        "surrogate_retained_capacity_max": 0.0,
        "surrogate_retained_capacity_mean": 0.0,
        "n_basis": int(len(basis)),
        "truncated": bool(getattr(basis, "truncated", False)),
    }


def _score_bundle_ipc_full(
    bundle: FeatureBundle,
    inputs: np.ndarray,
    basis: Sequence[ipc.BasisSpec],
    splits,
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
    raw_total = float(result["raw_total_ipc"])
    rank = int(result["effective_rank"])
    rank_fraction = float(raw_total / rank) if rank > 0 else float("inf")
    return {
        "ipc_total": float(result["total_ipc"]),
        "ipc_mc": float(result["mc"]),
        "ipc_nc": float(result["nc"]),
        "raw_total_ipc": raw_total,
        "effective_rank": rank,
        "rank_clamp_fraction": rank_fraction,
        "n_significant": int(result["n_significant"]),
        "threshold_saturated": bool(result["threshold_saturated"]),
        "threshold": float(result["threshold"]),
        "basis_edge_capacity": float(result["basis_edge_capacity"]),
        "surrogate_retained_capacity_max": float(
            result["surrogate_retained_capacity_max"]
        ),
        "surrogate_retained_capacity_mean": float(
            result["surrogate_retained_capacity_mean"]
        ),
        "n_basis": int(result["n_basis"]),
        "truncated": bool(result["truncated"]),
    }


def _feature_cache_ids(
    *,
    T: int,
    cell_id: str,
    window_k: int,
    gamma: float,
    dissipator: str,
) -> tuple[str, str]:
    gamma_token = run_stage3._float_token(float(gamma))
    prefix = f"ipc_pilot_T{int(T)}_{cell_id}_gamma{gamma_token}_diss{dissipator}"
    return prefix, f"{prefix}_k{int(window_k):02d}"


def _score_one_item(
    item: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    out_dir: Path,
    window_grid: Sequence[int],
    basis: Sequence[ipc.BasisSpec],
    basis_caps: Mapping[str, int],
    qrc_score_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cell: SeedCell = item["seed_cell"]
    input_seed = int(cell.input_seed)
    coupling_seed = int(cell.coupling_seed)
    cell_id = str(item["cell_id"])
    window_k = int(item["window_k"])
    gamma = float(item["gamma"])
    dissipator = str(item["dissipator"])
    T = int(config["T"])
    max_window_k = max(int(k) for k in window_grid)
    if int(basis_caps["max_delay"]) < max_window_k - 1:
        raise ValueError("IPC basis max_delay must cover the largest window first_valid")

    rng = np.random.default_rng(input_seed)
    inputs = rng.uniform(0.0, 1.0, size=T)
    basis_targets = {spec.key: ipc.basis_target(inputs, spec) for spec in basis}
    alphas = tuple(float(alpha) for alpha in config["hp_grids"]["alpha"])
    n_surrogates = int(config["ipc"].get("n_surrogates", ipc.DEFAULT_N_SURROGATES))

    structural = run_stage3._fixed_configs(
        config,
        gamma=gamma,
        dissipator=dissipator,
    )
    qrc_cfg: QRCConfig = structural["qrc"]
    qelm_cfg = windowed_qelm_from_qrc(qrc_cfg, window_k=window_k)
    observables = run_stage1._observables_from_config(config)
    assert_matched(qrc_cfg, qelm_cfg, observables=observables)

    qrc_cache_id, qelm_cache_id = _feature_cache_ids(
        T=T,
        cell_id=cell_id,
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
    splits = run_stage_ipc._shared_ipc_splits(
        qrc_bundle,
        qelm_bundle,
        targets=basis_targets,
        washout=int(config["washout"]),
        basis_max_delay=int(basis_caps["max_delay"]),
        max_window_k=max_window_k,
    )
    threshold_seed = run_stage_ipc._stable_seed(
        "ipc-pilot-threshold",
        T,
        input_seed,
        coupling_seed,
        run_stage3._float_token(gamma),
        dissipator,
    )

    qrc_key = (
        f"T{T}_{cell_id}_gamma{run_stage3._float_token(gamma)}_"
        f"diss{dissipator}_maxk{max_window_k}_threshold{threshold_seed}"
    )
    qrc_score = qrc_score_cache.get(qrc_key)
    if qrc_score is None:
        qrc_score = _score_bundle_ipc_full(
            qrc_bundle,
            inputs,
            basis,
            splits,
            alphas=alphas,
            n_surrogates=n_surrogates,
            rng_seed=threshold_seed,
            features_dead=bool(liveness["qrc"]["features_dead"]),
        )
        qrc_score_cache[qrc_key] = qrc_score
    qelm_score = _score_bundle_ipc_full(
        qelm_bundle,
        inputs,
        basis,
        splits,
        alphas=alphas,
        n_surrogates=n_surrogates,
        rng_seed=threshold_seed,
        features_dead=bool(liveness["qelm"]["features_dead"]),
    )

    common = {
        "T": T,
        "washout": int(config["washout"]),
        "cell_id": cell_id,
        "input_seed": input_seed,
        "coupling_seed": coupling_seed,
        "shot_seed": int(cell.shot_seed),
        "window_k": window_k,
        "gamma": gamma,
        "dissipator": dissipator,
        "n_qubits": int(qrc_cfg.n_qubits),
        "D": int(qrc_bundle.features.shape[1]),
        "n_surrogates": n_surrogates,
        "basis_max_delay": int(basis_caps["max_delay"]),
        "basis_max_degree": int(basis_caps["max_degree"]),
        "basis_max_total_degree": int(basis_caps["max_total_degree"]),
        "basis_max_functions": int(basis_caps["max_functions"]),
        "split_rows": {
            "train": int(splits.train.size),
            "val": int(splits.val.size),
            "test": int(splits.test.size),
            "first": int(splits.all[0]),
            "last": int(splits.all[-1]),
        },
    }
    rows: list[dict[str, Any]] = []
    for model, score in (("qrc", qrc_score), ("qelm", qelm_score)):
        rows.append(
            {
                **common,
                "model": model,
                "features_dead": bool(liveness[model]["features_dead"]),
                "liveness_failure": liveness[model]["liveness_failure"],
                **score,
            }
        )
    return rows


def _score_grid(
    *,
    base_config: Mapping[str, Any],
    out_dir: Path,
    cells: Sequence[Mapping[str, Any]],
    t_value: int,
    windows: Sequence[int],
    gammas: Sequence[float],
    profile: IpcPilotProfile,
) -> list[dict[str, Any]]:
    washout = _washout_for_T(base_config, int(t_value))
    config = _config_for_T(base_config, T=int(t_value), washout=washout, profile=profile)
    basis, basis_caps = _basis_from_config(config)
    qrc_score_cache: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for gamma in gammas:
        for window_k in windows:
            for cell in cells:
                records.extend(
                    _score_one_item(
                        {
                            **cell,
                            "window_k": int(window_k),
                            "gamma": float(gamma),
                            "dissipator": PILOT_DISSIPATOR,
                        },
                        config=config,
                        out_dir=out_dir,
                        window_grid=windows,
                        basis=basis,
                        basis_caps=basis_caps,
                        qrc_score_cache=qrc_score_cache,
                    )
                )
    return records


def _mean_ci(values: Sequence[float], *, seed: int, n_boot: int) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    lo, hi = stats.bootstrap_ci(arr, seed=int(seed), n_boot=int(n_boot))
    return {"mean": float(np.mean(arr)), "ci_low": float(lo), "ci_high": float(hi)}


def _aggregate_metric(group: pd.DataFrame, *, seed: int, n_boot: int) -> dict[str, Any]:
    totals = group["ipc_total"].to_numpy(dtype=float)
    raw = group["raw_total_ipc"].to_numpy(dtype=float)
    ranks = group["effective_rank"].to_numpy(dtype=float)
    rank_fraction = np.divide(
        raw,
        ranks,
        out=np.full_like(raw, np.inf, dtype=float),
        where=ranks > 0,
    )
    total_ci = _mean_ci(totals, seed=seed, n_boot=n_boot)
    return {
        "ipc_total_mean": total_ci["mean"],
        "ipc_total_ci_low": total_ci["ci_low"],
        "ipc_total_ci_high": total_ci["ci_high"],
        "raw_total_mean": float(np.mean(raw)) if raw.size else float("nan"),
        "effective_rank_mean": float(np.mean(ranks)) if ranks.size else float("nan"),
        "rank_clamp_fraction_mean": float(np.mean(rank_fraction))
        if rank_fraction.size
        else float("nan"),
        "rank_clamp_fraction_max": float(np.max(rank_fraction))
        if rank_fraction.size
        else float("nan"),
        "n_significant_mean": float(group["n_significant"].mean())
        if len(group)
        else float("nan"),
        "basis_edge_capacity_mean": float(group["basis_edge_capacity"].mean())
        if len(group)
        else float("nan"),
        "basis_edge_capacity_max": float(group["basis_edge_capacity"].max())
        if len(group)
        else float("nan"),
        "threshold_saturated_fraction": float(group["threshold_saturated"].mean())
        if len(group)
        else float("nan"),
        "n": int(len(group)),
    }


def _reliability_series(score_frame: pd.DataFrame, *, n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if score_frame.empty:
        return pd.DataFrame(rows)

    qrc = (
        score_frame[score_frame["model"] == "qrc"]
        .drop_duplicates(["T", "cell_id", "gamma", "dissipator"])
        .copy()
    )
    for pos, (T, group) in enumerate(qrc.groupby("T", sort=True)):
        rows.append(
            {
                "T": int(T),
                "series_id": "QRC",
                "model": "qrc",
                "window_k": None,
                **_aggregate_metric(group, seed=BOOTSTRAP_SEED + pos, n_boot=n_boot),
            }
        )

    qelm = score_frame[score_frame["model"] == "qelm"].copy()
    for pos, ((T, window_k), group) in enumerate(qelm.groupby(["T", "window_k"], sort=True)):
        rows.append(
            {
                "T": int(T),
                "series_id": f"QELM(k={int(window_k)})",
                "model": "qelm",
                "window_k": int(window_k),
                **_aggregate_metric(
                    group,
                    seed=BOOTSTRAP_SEED + 10_000 + pos,
                    n_boot=n_boot,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["T", "series_id"]).reset_index(drop=True)


def select_stabilized_T(
    series: pd.DataFrame,
    *,
    tolerance: float = 0.05,
    rank_clamp_fraction_max: float = 0.98,
    basis_edge_max: float = SMALL_EDGE_CAPACITY,
) -> dict[str, Any]:
    """Pick the smallest T whose IPC estimates are stable and not clamped."""

    if series.empty:
        return {
            "t_star": None,
            "found": False,
            "reason": "no reliability series",
            "candidates": [],
        }

    candidates: list[dict[str, Any]] = []
    T_values = sorted(int(T) for T in series["T"].unique())
    for idx, T in enumerate(T_values):
        current = series[series["T"] == T].set_index("series_id")
        if idx == 0:
            candidates.append(
                {
                    "T": T,
                    "eligible": False,
                    "reason": "first T has no previous T for stability check",
                    "max_successive_relative_change": None,
                    "rank_clamp_ok": bool(
                        current["rank_clamp_fraction_max"].max()
                        < float(rank_clamp_fraction_max)
                    ),
                    "basis_edge_ok": bool(
                        current["basis_edge_capacity_max"].max() <= float(basis_edge_max)
                    ),
                }
            )
            continue
        previous = series[series["T"] == T_values[idx - 1]].set_index("series_id")
        common = sorted(set(current.index) & set(previous.index))
        rel_changes: list[float] = []
        for series_id in common:
            now = float(current.loc[series_id, "ipc_total_mean"])
            before = float(previous.loc[series_id, "ipc_total_mean"])
            scale = max(abs(now), abs(before), 1.0e-12)
            rel_changes.append(abs(now - before) / scale)
        max_rel_change = float(np.max(rel_changes)) if rel_changes else float("inf")
        rank_clamp_ok = bool(
            current["rank_clamp_fraction_max"].max() < float(rank_clamp_fraction_max)
        )
        basis_edge_ok = bool(
            current["basis_edge_capacity_max"].max() <= float(basis_edge_max)
        )
        stable = bool(max_rel_change < float(tolerance))
        eligible = bool(stable and rank_clamp_ok and basis_edge_ok)
        reason = "PASS" if eligible else "not stable/clamp-free/basis-edge-small"
        candidates.append(
            {
                "T": T,
                "eligible": eligible,
                "reason": reason,
                "max_successive_relative_change": max_rel_change,
                "rank_clamp_ok": rank_clamp_ok,
                "basis_edge_ok": basis_edge_ok,
                "rank_clamp_fraction_max": float(
                    current["rank_clamp_fraction_max"].max()
                ),
                "basis_edge_capacity_max": float(
                    current["basis_edge_capacity_max"].max()
                ),
            }
        )
        if eligible:
            return {
                "t_star": T,
                "found": True,
                "reason": "smallest T satisfying stability, unclamped rank, and small basis edge",
                "candidates": candidates,
            }

    return {
        "t_star": None,
        "found": False,
        "reason": "no T satisfied stability, unclamped rank, and small basis edge",
        "candidates": candidates,
    }


def summarize_t_reliability(
    score_frame: pd.DataFrame,
    *,
    profile: IpcPilotProfile,
) -> dict[str, Any]:
    series = _reliability_series(score_frame, n_boot=profile.n_boot)
    selection = select_stabilized_T(
        series,
        tolerance=profile.stabilization_tol,
        rank_clamp_fraction_max=profile.rank_clamp_fraction_max,
        basis_edge_max=profile.basis_edge_max,
    )
    selection_by_T = {int(row["T"]): row for row in selection["candidates"]}
    by_T: list[dict[str, Any]] = []
    for T in sorted(int(value) for value in series["T"].unique()):
        current = series[series["T"] == T]
        qrc_row = current[current["series_id"] == "QRC"]
        qelm_rows = current[current["model"] == "qelm"].sort_values("window_k")
        by_T.append(
            {
                "T": T,
                "qrc": qrc_row.iloc[0].to_dict() if not qrc_row.empty else {},
                "qelm_by_k": {
                    str(int(row.window_k)): row._asdict()
                    for row in qelm_rows.itertuples(index=False)
                },
                "selection_diagnostic": selection_by_T.get(T, {}),
            }
        )
    return {
        "summary_by_T": by_T,
        "series": series.to_dict(orient="records"),
        "selection": selection,
        "criteria": {
            "successive_T_relative_change_lt": float(profile.stabilization_tol),
            "rank_clamp_fraction_max_lt": float(profile.rank_clamp_fraction_max),
            "basis_edge_capacity_max_lte": float(profile.basis_edge_max),
        },
    }


def _surface_pairs(score_frame: pd.DataFrame) -> pd.DataFrame:
    pivot = score_frame.pivot_table(
        index=["gamma", "window_k", "cell_id", "input_seed", "coupling_seed"],
        columns="model",
        values="ipc_total",
        aggfunc="first",
    ).reset_index()
    if "qrc" not in pivot.columns or "qelm" not in pivot.columns:
        return pd.DataFrame()
    pivot["delta"] = pivot["qrc"] - pivot["qelm"]
    return pivot.rename(columns={"qrc": "ipc_qrc", "qelm": "ipc_qelm"})


def _delta_label(delta: float, ci_low: float, ci_high: float) -> str:
    if np.isfinite([delta, ci_low, ci_high]).all():
        if ci_low > 0.0:
            return "QRC"
        if ci_high < 0.0:
            return "QELM"
    if float(delta) > 0.0:
        return "QRC_point"
    if float(delta) < 0.0:
        return "QELM_point"
    return "tie"


def _paired_surface_stats(
    pairs: pd.DataFrame,
    *,
    n_boot: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pos, ((gamma, window_k), group) in enumerate(
        pairs.groupby(["gamma", "window_k"], sort=True)
    ):
        delta_stats = stats.paired_delta_stats(
            group["ipc_qrc"].to_numpy(dtype=float),
            group["ipc_qelm"].to_numpy(dtype=float),
            seed=BOOTSTRAP_SEED + 20_000 + pos,
            n_boot=n_boot,
        )
        rows.append(
            {
                "gamma": float(gamma),
                "window_k": int(window_k),
                "ipc_qrc_mean": float(group["ipc_qrc"].mean()),
                "ipc_qelm_mean": float(group["ipc_qelm"].mean()),
                "delta_mean": float(delta_stats["mean_delta"]),
                "delta_ci_low": float(delta_stats["ci_low"]),
                "delta_ci_high": float(delta_stats["ci_high"]),
                "delta_std": float(delta_stats["delta_std"]),
                "delta_se": float(delta_stats["se"]),
                "n_pairs": int(delta_stats["n_pairs"]),
                "win_rate_qrc": float(delta_stats["win_rate"]),
                "p_wilcoxon": float(delta_stats["p_wilcoxon"]),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["label"] = [
        _delta_label(row.delta_mean, row.delta_ci_low, row.delta_ci_high)
        for row in table.itertuples(index=False)
    ]
    return table.sort_values(["gamma", "window_k"]).reset_index(drop=True)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x_arr, y_arr).statistic)
    except Exception:
        return float(pd.Series(x_arr).rank().corr(pd.Series(y_arr).rank()))


def _crossing_from_points(gammas: Sequence[float], deltas: Sequence[float]) -> float | None:
    xs = np.asarray(gammas, dtype=float)
    ys = np.asarray(deltas, dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    for idx in range(len(xs) - 1):
        y0 = float(ys[idx])
        y1 = float(ys[idx + 1])
        if y0 == 0.0:
            return float(xs[idx])
        if y0 * y1 < 0.0:
            denom = y1 - y0
            if abs(denom) <= 1.0e-15:
                return float(xs[idx])
            return float(xs[idx] + (0.0 - y0) * (xs[idx + 1] - xs[idx]) / denom)
    if len(xs) and float(ys[-1]) == 0.0:
        return float(xs[-1])
    return None


def detect_sign_crossover(delta_infty: pd.DataFrame) -> dict[str, Any]:
    """Detect whether Delta_infty(gamma) changes sign."""

    if delta_infty.empty:
        return {
            "both_signs_present": False,
            "gamma_star": None,
            "crossings": [],
            "trend_spearman": float("nan"),
        }
    ordered = delta_infty.sort_values("gamma")
    gammas = ordered["gamma"].to_numpy(dtype=float)
    deltas = ordered["delta_mean"].to_numpy(dtype=float)
    finite = np.isfinite(deltas)
    both_signs = bool(np.any(deltas[finite] > 0.0) and np.any(deltas[finite] < 0.0))
    crossings: list[float] = []
    for idx in range(len(gammas) - 1):
        crossing = _crossing_from_points(gammas[idx : idx + 2], deltas[idx : idx + 2])
        if crossing is not None:
            crossings.append(float(crossing))
    return {
        "both_signs_present": both_signs,
        "gamma_star": crossings[0] if crossings else None,
        "crossings": crossings,
        "trend_spearman": _spearman(gammas, deltas),
        "delta_infty_by_gamma": ordered.to_dict(orient="records"),
    }


def _bootstrap_gamma_star(
    pairs: pd.DataFrame,
    *,
    max_window: int,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    largest = pairs[pairs["window_k"] == int(max_window)]
    if largest.empty:
        return {
            "gamma_star_ci_low": None,
            "gamma_star_ci_high": None,
            "trend_spearman_ci_low": None,
            "trend_spearman_ci_high": None,
            "n_boot_with_crossing": 0,
        }
    pivot = largest.pivot_table(
        index="cell_id",
        columns="gamma",
        values="delta",
        aggfunc="first",
    )
    gammas = np.asarray(sorted(float(col) for col in pivot.columns), dtype=float)
    matrix = pivot.loc[:, gammas].to_numpy(dtype=float)
    n_cells = matrix.shape[0]
    if n_cells <= 1:
        means = np.nanmean(matrix, axis=0)
        crossing = _crossing_from_points(gammas, means)
        rho = _spearman(gammas, means)
        return {
            "gamma_star_ci_low": crossing,
            "gamma_star_ci_high": crossing,
            "trend_spearman_ci_low": rho,
            "trend_spearman_ci_high": rho,
            "n_boot_with_crossing": int(crossing is not None),
        }

    rng = np.random.default_rng(int(seed))
    crossings: list[float] = []
    rhos: list[float] = []
    for _ in range(int(n_boot)):
        sample = rng.integers(0, n_cells, n_cells)
        means = np.nanmean(matrix[sample], axis=0)
        crossing = _crossing_from_points(gammas, means)
        if crossing is not None:
            crossings.append(float(crossing))
        rho = _spearman(gammas, means)
        if np.isfinite(rho):
            rhos.append(float(rho))

    def ci(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        lo, hi = np.percentile(values, [2.5, 97.5])
        return float(lo), float(hi)

    gamma_lo, gamma_hi = ci(crossings)
    rho_lo, rho_hi = ci(rhos)
    return {
        "gamma_star_ci_low": gamma_lo,
        "gamma_star_ci_high": gamma_hi,
        "trend_spearman_ci_low": rho_lo,
        "trend_spearman_ci_high": rho_hi,
        "n_boot_with_crossing": int(len(crossings)),
    }


def _bootstrap_qelm_k_spearman(
    pairs: pd.DataFrame,
    *,
    gamma: float,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    subset = pairs[pairs["gamma"] == float(gamma)]
    if subset.empty:
        return {"rho": float("nan"), "ci_low": None, "ci_high": None}
    pivot = subset.pivot_table(
        index="cell_id",
        columns="window_k",
        values="ipc_qelm",
        aggfunc="first",
    )
    windows = np.asarray(sorted(int(col) for col in pivot.columns), dtype=float)
    matrix = pivot.loc[:, windows].to_numpy(dtype=float)
    means = np.nanmean(matrix, axis=0)
    rho = _spearman(windows, means)
    if matrix.shape[0] <= 1:
        return {"rho": rho, "ci_low": rho, "ci_high": rho}

    rng = np.random.default_rng(int(seed))
    rhos: list[float] = []
    for _ in range(int(n_boot)):
        sample = rng.integers(0, matrix.shape[0], matrix.shape[0])
        sample_means = np.nanmean(matrix[sample], axis=0)
        sample_rho = _spearman(windows, sample_means)
        if np.isfinite(sample_rho):
            rhos.append(float(sample_rho))
    if not rhos:
        return {"rho": rho, "ci_low": None, "ci_high": None}
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return {"rho": rho, "ci_low": float(lo), "ci_high": float(hi)}


def _convergence_by_gamma(
    table: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    n_boot: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pos, (gamma, group) in enumerate(table.groupby("gamma", sort=True)):
        ordered = group.sort_values("window_k")
        windows = ordered["window_k"].to_numpy(dtype=int)
        deltas = ordered["delta_mean"].to_numpy(dtype=float)
        qelm = ordered["ipc_qelm_mean"].to_numpy(dtype=float)
        delta_infty = float(deltas[-1])
        distances = np.abs(deltas - delta_infty)
        tolerance = max(0.02, 0.05 * max(abs(delta_infty), 1.0e-12))
        eligible = np.flatnonzero(distances <= tolerance)
        k_star = int(windows[int(eligible[0])]) if eligible.size else None
        qelm_spearman = _bootstrap_qelm_k_spearman(
            pairs,
            gamma=float(gamma),
            seed=BOOTSTRAP_SEED + 30_000 + pos,
            n_boot=n_boot,
        )
        out.append(
            {
                "gamma": float(gamma),
                "delta_infty": delta_infty,
                "k_star_within_5pct_or_0p02": k_star,
                "qelm_non_decreasing": bool(np.all(np.diff(qelm) >= -1.0e-9)),
                "delta_distance_to_infty_non_increasing": bool(
                    np.all(np.diff(distances) <= 1.0e-9)
                ),
                "qelm_spearman_k": qelm_spearman,
                "delta_spearman_k": _spearman(windows, deltas),
                "distance_to_delta_infty_by_k": {
                    str(int(k)): float(distance)
                    for k, distance in zip(windows, distances, strict=True)
                },
            }
        )
    return out


def _reservoir_for_gap(qrc_cfg: QRCConfig, coupling_seed: int):
    rng = np.random.default_rng(int(coupling_seed))
    J = reservoirs.random_couplings(qrc_cfg.n_qubits, qrc_cfg.J_s, rng)
    H_static = reservoirs.ising_xx_hamiltonian(J, qrc_cfg.h, qrc_cfg.n_qubits)
    H_drive = qrc_cfg.h_in * reservoirs.transverse_drive(qrc_cfg.n_qubits)
    jumps = _dissipator_jumps(qrc_cfg.dissipator, qrc_cfg.n_qubits, qrc_cfg.gamma)
    return reservoirs.LindbladReservoir.from_terms(
        qrc_cfg.n_qubits,
        H_static,
        H_drive,
        jumps,
        qrc_cfg.dt,
        cache_propagators=False,
        quantize=None,
    )


def _spectral_gap_by_gamma(
    *,
    base_config: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    gammas: Sequence[float],
    n_gap_couplings: int,
    s_value: float = 0.5,
) -> list[dict[str, Any]]:
    unique_couplings = []
    for cell in cells:
        coupling = int(cell["seed_cell"].coupling_seed)
        if coupling not in unique_couplings:
            unique_couplings.append(coupling)
    unique_couplings = unique_couplings[: int(n_gap_couplings)]
    rows: list[dict[str, Any]] = []
    for gamma in gammas:
        qrc_cfg = run_stage3._fixed_configs(
            base_config,
            gamma=float(gamma),
            dissipator=PILOT_DISSIPATOR,
        )["qrc"]
        gaps: list[float] = []
        errors: list[str] = []
        for coupling_seed in unique_couplings:
            try:
                reservoir = _reservoir_for_gap(qrc_cfg, coupling_seed)
                gap = diagnostics.spectral_gap(reservoir.liouvillian(float(s_value)))
                gaps.append(float(gap))
            except Exception as exc:
                errors.append(f"coupling_seed={coupling_seed}: {type(exc).__name__}: {exc}")
        rows.append(
            {
                "gamma": float(gamma),
                "s_value": float(s_value),
                "n_couplings": int(len(unique_couplings)),
                "coupling_seeds": [int(seed) for seed in unique_couplings],
                "spectral_gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
                "spectral_gap_median": float(np.median(gaps)) if gaps else float("nan"),
                "spectral_gap_values": gaps,
                "errors": errors,
            }
        )
    return rows


def _relate_gap_to_window(
    spectral_gaps: Sequence[Mapping[str, Any]],
    convergence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gap_by_gamma = {
        float(row["gamma"]): float(row["spectral_gap_mean"])
        for row in spectral_gaps
        if np.isfinite(float(row.get("spectral_gap_mean", np.nan)))
    }
    xs: list[float] = []
    ys: list[float] = []
    paired: list[dict[str, Any]] = []
    for row in convergence:
        gamma = float(row["gamma"])
        k_star = row.get("k_star_within_5pct_or_0p02")
        gap = gap_by_gamma.get(gamma)
        if gap is None or k_star is None:
            continue
        xs.append(gap)
        ys.append(float(k_star))
        paired.append({"gamma": gamma, "spectral_gap_mean": gap, "k_star": int(k_star)})
    return {
        "paired_gamma_gap_k_star": paired,
        "spearman_gap_vs_k_star": _spearman(xs, ys) if len(xs) >= 2 else float("nan"),
        "interpretation": (
            "larger spectral gap should correspond to shorter required QELM window "
            "if the window length is tracking reservoir memory"
        ),
    }


def summarize_signed_surface(
    score_frame: pd.DataFrame,
    *,
    windows: Sequence[int],
    n_boot: int,
    spectral_gaps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pairs = _surface_pairs(score_frame)
    table = _paired_surface_stats(pairs, n_boot=n_boot)
    max_window = max(int(k) for k in windows)
    delta_infty = table[table["window_k"] == max_window].copy()
    crossover = detect_sign_crossover(delta_infty)
    crossover.update(
        _bootstrap_gamma_star(
            pairs,
            max_window=max_window,
            seed=BOOTSTRAP_SEED + 40_000,
            n_boot=n_boot,
        )
    )
    convergence = _convergence_by_gamma(table, pairs, n_boot=n_boot)
    return {
        "signed_delta_table": table.to_dict(orient="records"),
        "delta_infty": delta_infty.to_dict(orient="records"),
        "max_window_for_delta_infty": int(max_window),
        "sign_structure": crossover,
        "convergence_by_gamma": convergence,
        "spectral_gap": list(spectral_gaps),
        "spectral_gap_relation": _relate_gap_to_window(spectral_gaps, convergence),
        "per_seed_pairs": pairs.to_dict(orient="records"),
    }


def _ci_excludes_zero(row: Mapping[str, Any]) -> bool:
    lo = float(row["delta_ci_low"])
    hi = float(row["delta_ci_high"])
    return bool(np.isfinite([lo, hi]).all() and (lo > 0.0 or hi < 0.0))


def _nearest_gamma_row(rows: Sequence[Mapping[str, Any]], gamma: float) -> Mapping[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row["gamma"]) - float(gamma)))


def propose_signed_gates(
    *,
    part_a: Mapping[str, Any],
    part_b: Mapping[str, Any],
    analysis_T: int,
    profile: IpcPilotProfile,
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    delta_infty = list(part_b["delta_infty"])
    tuned_row = _nearest_gamma_row(delta_infty, TUNED_GAMMA)
    if tuned_row is None:
        gi2_verdict = "MISSING_TUNED_GAMMA"
        tuned_persistent = False
        gi2_resolvable = False
    else:
        lo = float(tuned_row["delta_ci_low"])
        hi = float(tuned_row["delta_ci_high"])
        mean = float(tuned_row["delta_mean"])
        if lo > 0.0:
            gi2_verdict = "PERSISTENT_GAP"
            tuned_persistent = True
            gi2_resolvable = True
        elif lo <= 0.0 <= hi:
            gi2_verdict = "QELM_CONVERGES"
            tuned_persistent = False
            gi2_resolvable = False
        elif hi < 0.0:
            gi2_verdict = "WINDOW_WINS_AT_TUNED_GAMMA"
            tuned_persistent = False
            gi2_resolvable = True
        else:
            gi2_verdict = "UNRESOLVED"
            tuned_persistent = False
            gi2_resolvable = False
        tuned_row = {**tuned_row, "nearest_to_gamma": TUNED_GAMMA, "delta_mean": mean}

    gi1_by_gamma = []
    for row in part_b["convergence_by_gamma"]:
        qelm_spearman = row["qelm_spearman_k"]
        ci_low = qelm_spearman.get("ci_low")
        passed = bool(
            row["qelm_non_decreasing"]
            and ci_low is not None
            and float(ci_low) >= 0.0
        )
        gi1_by_gamma.append(
            {
                "gamma": float(row["gamma"]),
                "proposed_pass": passed,
                "qelm_non_decreasing": bool(row["qelm_non_decreasing"]),
                "qelm_spearman_k": qelm_spearman,
                "k_star_within_5pct_or_0p02": row["k_star_within_5pct_or_0p02"],
            }
        )

    sign = part_b["sign_structure"]
    both_signs = bool(sign["both_signs_present"])
    t_star_found = bool(part_a["selection"]["found"])
    diagnostics_list: list[str] = []
    if not t_star_found:
        diagnostics_list.append(
            "T* not found: inspect rank clamp fractions, basis edge capacity, and "
            "extend T or IPC basis before confirmatory freeze"
        )
    if not tuned_persistent:
        diagnostics_list.append(
            f"GI2 tuned-gamma persistent gap not established: {gi2_verdict}"
        )
    if not both_signs:
        diagnostics_list.append(
            "signed surface is degenerate across gamma: Delta_infty does not show both signs"
        )
    go = bool(t_star_found and tuned_persistent and both_signs)
    basis_caps = run_stage_ipc._basis_caps(
        _config_for_T(
            base_config,
            T=int(analysis_T),
            washout=_washout_for_T(base_config, int(analysis_T)),
            profile=profile,
        )
    )
    return {
        "proposed_gates": {
            "GI1": {
                "description": (
                    "per-gamma IPC(QELM,k) is significantly non-decreasing in k "
                    "toward Delta_infty"
                ),
                "method": "Spearman over k with bootstrap CI over pilot seed cells",
                "by_gamma": gi1_by_gamma,
            },
            "GI2": {
                "description": (
                    "at tuned gamma approx 0.3, Delta_infty > 0 with bootstrap CI "
                    "excluding 0 implies PERSISTENT_GAP; CI including 0 implies "
                    "QELM_CONVERGES"
                ),
                "verdict": gi2_verdict,
                "resolvable": gi2_resolvable,
                "tuned_gamma_row": tuned_row,
            },
            "GI3": {
                "description": (
                    "Delta_infty(gamma) changes sign at gamma* and gamma* relates "
                    "to the spectral gap"
                ),
                "both_signs_present": both_signs,
                "gamma_star": sign["gamma_star"],
                "gamma_star_ci": [
                    sign.get("gamma_star_ci_low"),
                    sign.get("gamma_star_ci_high"),
                ],
                "trend_spearman": sign["trend_spearman"],
                "trend_spearman_ci": [
                    sign.get("trend_spearman_ci_low"),
                    sign.get("trend_spearman_ci_high"),
                ],
                "spectral_gap_relation": part_b["spectral_gap_relation"],
            },
        },
        "go_no_go": {
            "status": "GO" if go else "NO-GO",
            "go": go,
            "criteria": {
                "T_star_found": t_star_found,
                "tuned_gamma_persistent_gap_measurable": tuned_persistent,
                "signed_surface_non_degenerate_both_signs": both_signs,
            },
            "diagnostics": diagnostics_list,
        },
        "proposed_confirmatory_grid": {
            "T_star": part_a["selection"]["t_star"],
            "analysis_T_used_in_part_B": int(analysis_T),
            "window_grid": [int(k) for k in profile.surface_windows],
            "gamma_grid": [float(gamma) for gamma in profile.gamma_grid],
            "dissipator": PILOT_DISSIPATOR,
            "n_qubits": int(base_config["primary"]["n_qubits"]),
            "ipc_basis": {key: int(value) for key, value in basis_caps.items()},
            "seed_note": "pilot grid only; confirmatory seeds untouched by this script",
        },
    }


def run_ipc_pilot(
    profile_name: str,
    out_dir: Path,
    *,
    t_values: tuple[int, ...] | None = None,
    reliability_windows: tuple[int, ...] | None = None,
    surface_windows: tuple[int, ...] | None = None,
    gammas: tuple[float, ...] | None = None,
    skip_spectral_gap: bool = False,
) -> dict[str, Any]:
    base_profile = PROFILES[profile_name]
    profile = _profile_with_overrides(
        base_profile,
        t_values=t_values,
        reliability_windows=reliability_windows,
        surface_windows=surface_windows,
        gammas=gammas,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    base_config = run_stage1._load_config(profile_name)
    cells = _pilot_cells(profile.max_seed_cells)
    reliability_records: list[dict[str, Any]] = []
    for T in profile.t_values:
        reliability_records.extend(
            _score_grid(
                base_config=base_config,
                out_dir=out_dir,
                cells=cells,
                t_value=int(T),
                windows=profile.reliability_windows,
                gammas=(TUNED_GAMMA,),
                profile=profile,
            )
        )
    reliability_frame = pd.DataFrame(reliability_records)
    part_a = summarize_t_reliability(reliability_frame, profile=profile)
    t_star = part_a["selection"]["t_star"]
    analysis_T = int(t_star) if t_star is not None else int(max(profile.t_values))

    surface_records = _score_grid(
        base_config=base_config,
        out_dir=out_dir,
        cells=cells,
        t_value=analysis_T,
        windows=profile.surface_windows,
        gammas=profile.gamma_grid,
        profile=profile,
    )
    surface_frame = pd.DataFrame(surface_records)
    spectral_gaps = (
        []
        if skip_spectral_gap
        else _spectral_gap_by_gamma(
            base_config=base_config,
            cells=cells,
            gammas=profile.gamma_grid,
            n_gap_couplings=profile.n_gap_couplings,
        )
    )
    part_b = summarize_signed_surface(
        surface_frame,
        windows=profile.surface_windows,
        n_boot=profile.n_boot,
        spectral_gaps=spectral_gaps,
    )
    part_c = propose_signed_gates(
        part_a=part_a,
        part_b=part_b,
        analysis_T=analysis_T,
        profile=profile,
        base_config=base_config,
    )

    result = {
        "profile": profile_name,
        "profile_settings": asdict(profile),
        "pilot_only": True,
        "confirmatory_seeds_touched": False,
        "seed_grid": grid_manifest(PILOT_SEED_GRID),
        "seed_cells_used": [
            {
                "cell_id": str(cell["cell_id"]),
                "input_seed": int(cell["seed_cell"].input_seed),
                "coupling_seed": int(cell["seed_cell"].coupling_seed),
                "shot_seed": int(cell["seed_cell"].shot_seed),
            }
            for cell in cells
        ],
        "config": {
            "base_profile": profile_name,
            "n_qubits": int(base_config["primary"]["n_qubits"]),
            "dissipator": PILOT_DISSIPATOR,
            "tuned_gamma": TUNED_GAMMA,
            "washout_rule": "round(T * config.washout / config.T)",
            "signed_delta": "IPC(QRC) - IPC(QELM,k)",
        },
        "part_a": part_a,
        "part_b": part_b,
        "part_c": part_c,
        "runtime": {"elapsed_s": float(time.perf_counter() - start)},
    }

    out_path = out_dir / "pilot.json"
    out_path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _format_ci(row: Mapping[str, Any]) -> str:
    return (
        f"{float(row['delta_mean']):+.4f} "
        f"[{float(row['delta_ci_low']):+.4f}, {float(row['delta_ci_high']):+.4f}]"
    )


def _print_delta_table(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        print("signed Delta(k,gamma): no rows")
        return
    gammas = sorted({float(row["gamma"]) for row in rows})
    windows = sorted({int(row["window_k"]) for row in rows})
    lookup = {(float(row["gamma"]), int(row["window_k"])): row for row in rows}
    header = "gamma " + " ".join(f"k={k}" for k in windows)
    print("signed Delta(k,gamma) = IPC(QRC) - IPC(QELM,k)")
    print(header)
    for gamma in gammas:
        values = []
        for window in windows:
            row = lookup.get((gamma, window))
            values.append("NA" if row is None else f"{float(row['delta_mean']):+.4f}")
        print(f"{gamma:.3g} " + " ".join(values))


def print_summary(result: Mapping[str, Any]) -> None:
    part_a = result["part_a"]
    part_b = result["part_b"]
    part_c = result["part_c"]
    selection = part_a["selection"]
    status = part_c["go_no_go"]["status"]
    t_star = selection["t_star"]
    analysis_T = part_c["proposed_confirmatory_grid"]["analysis_T_used_in_part_B"]
    print(f"ipc_pilot profile={result['profile']} status={status}")
    print(f"T_star={t_star if t_star is not None else 'NONE'} analysis_T={analysis_T}")
    if not selection["found"]:
        print(f"T_reliability: {selection['reason']}")
    for row in part_a["summary_by_T"]:
        diag = row["selection_diagnostic"]
        rel = diag.get("max_successive_relative_change")
        rel_text = "NA" if rel is None else f"{float(rel):.3f}"
        qrc = row["qrc"]
        print(
            "T={T} QRC={qrc:.4f} clamp_max={clamp:.3f} "
            "basis_edge_max={edge:.3f} rel_change={rel}".format(
                T=int(row["T"]),
                qrc=float(qrc.get("ipc_total_mean", float("nan"))),
                clamp=float(qrc.get("rank_clamp_fraction_max", float("nan"))),
                edge=float(qrc.get("basis_edge_capacity_max", float("nan"))),
                rel=rel_text,
            )
        )

    _print_delta_table(part_b["signed_delta_table"])
    print("Delta_infty by gamma:")
    for row in part_b["delta_infty"]:
        print(f"  gamma={float(row['gamma']):.3g} {_format_ci(row)}")
    sign = part_b["sign_structure"]
    print(
        "sign_crossover both_signs={both} gamma_star={gamma_star} "
        "trend_spearman={rho}".format(
            both=sign["both_signs_present"],
            gamma_star=sign["gamma_star"],
            rho=sign["trend_spearman"],
        )
    )
    gi2 = part_c["proposed_gates"]["GI2"]
    print(f"GI2 proposed verdict={gi2['verdict']}")
    print(
        "GO/NO-GO: {status}; confirmatory_grid T*={T} k={k} gamma={gamma}".format(
            status=status,
            T=part_c["proposed_confirmatory_grid"]["T_star"],
            k=part_c["proposed_confirmatory_grid"]["window_grid"],
            gamma=part_c["proposed_confirmatory_grid"]["gamma_grid"],
        )
    )
    for diagnostic in part_c["go_no_go"]["diagnostics"]:
        print(f"no_go_diagnostic: {diagnostic}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--t-values", type=_parse_int_csv, default=None)
    parser.add_argument("--reliability-windows", type=_parse_int_csv, default=None)
    parser.add_argument("--surface-windows", type=_parse_int_csv, default=None)
    parser.add_argument("--gammas", type=_parse_float_csv, default=None)
    parser.add_argument(
        "--skip-spectral-gap",
        action="store_true",
        help="skip dense Liouvillian eigenspectrum diagnostics",
    )
    args = parser.parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    result = run_ipc_pilot(
        str(args.profile),
        out_dir,
        t_values=args.t_values,
        reliability_windows=args.reliability_windows,
        surface_windows=args.surface_windows,
        gammas=args.gammas,
        skip_spectral_gap=bool(args.skip_spectral_gap),
    )
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
