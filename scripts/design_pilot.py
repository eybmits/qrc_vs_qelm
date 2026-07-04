#!/usr/bin/env python
"""Design-stage pilot for the QRC-vs-QELM phase diagram.

Pilot outputs are design diagnostics only. They must not be imported into the
confirmatory analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrc_vs_qelm import controls, pipeline, stats, task_families
from qrc_vs_qelm.gates import BOOTSTRAP_SEED, FDR_Q, STAGE1_MIN_REGIME_FRACTION
from qrc_vs_qelm.matching import (
    assert_bundles_matched,
    assert_features_live,
    assert_matched,
    recommend_T_evolve,
)
from qrc_vs_qelm.models.api import QELMConfig, QRCConfig, build_features
from qrc_vs_qelm.seeds import SEED_GRIDS, grid_manifest


DEFAULT_OUT = ROOT / "results_pilot"

TAUS = (0, 1, 2, 3, 5, 8)
DEGREES = (1, 2, 3, 5)
TIE_BANDS = (0.01, 0.02, 0.04)
PRIMARY_TIE_BAND = 0.02
ALPHAS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2)


@dataclass(frozen=True)
class PilotProfile:
    T: int
    washout: int
    n_boot: int
    gamma_slices: tuple[float, ...]


PROFILES = {
    "smoke": PilotProfile(T=180, washout=30, n_boot=256, gamma_slices=(0.0,)),
    "full": PilotProfile(T=480, washout=80, n_boot=2000, gamma_slices=(0.0, 0.1)),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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
        return tuple(int(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _target_name(tau: int, degree: int) -> str:
    return f"leg_tau{int(tau)}_p{int(degree)}"


def _base_configs(gamma: float) -> tuple[QRCConfig, QELMConfig]:
    qrc_cfg = QRCConfig(
        dt=0.5,
        h_in=0.5,
        n_virtual=1,
        n_qubits=5,
        J_s=1.0,
        h=0.5,
        gamma=float(gamma),
        dissipator="local_loss",
        quantize_levels=None,
    )
    qelm_base = QELMConfig(
        d=5,
        T_evolve=0.1,
        n_snapshots=1,
        n_qubits=5,
        J_s=1.0,
        h=0.5,
        gamma=float(gamma),
        dissipator="local_loss",
        quantize_levels=None,
    )
    return qrc_cfg, qelm_base


def _selected_configs(gamma: float, coupling_seed: int) -> tuple[QRCConfig, QELMConfig, dict]:
    qrc_cfg, qelm_base = _base_configs(gamma)
    recommendation = recommend_T_evolve(qelm_base, int(coupling_seed))
    qelm_cfg = replace(qelm_base, T_evolve=float(recommendation["t_recommended"]))
    assert_matched(qrc_cfg, qelm_cfg)
    return qrc_cfg, qelm_cfg, recommendation


def _evaluate_one_matrix(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    splits: pipeline.Splits,
    *,
    first_valid: int,
) -> pd.DataFrame:
    return pipeline.evaluate_feature_matrix(
        features,
        targets,
        splits,
        alphas=ALPHAS,
        first_valid=first_valid,
    )


def _score_seed_cell(
    *,
    input_seed: int,
    coupling_seed: int,
    gamma: float,
    profile: PilotProfile,
    taus: tuple[int, ...],
    degrees: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(int(input_seed))
    inputs = rng.uniform(0.0, 1.0, size=int(profile.T))
    targets = task_families.build_family_targets(inputs, taus=taus, degrees=degrees)
    qrc_cfg, qelm_cfg, recommendation = _selected_configs(gamma, int(coupling_seed))

    qrc_bundle = build_features(qrc_cfg, inputs, int(coupling_seed))
    qelm_bundle = build_features(qelm_cfg, inputs, int(coupling_seed))
    assert_bundles_matched(qrc_bundle, qelm_bundle)
    assert_features_live(qrc_bundle)
    assert_features_live(qelm_bundle)

    idx = pipeline.stage_valid_indices(
        [qrc_bundle, qelm_bundle],
        targets,
        washout=int(profile.washout),
        max_delay=max(taus),
    )
    splits = pipeline.make_splits(idx)
    D = int(qrc_bundle.features.shape[1])
    linear_features = controls.linear_delay_features(inputs, d=qelm_cfg.d, D=D)
    poly_features = controls.poly_delay_features(inputs, d=qelm_cfg.d, D=D)

    qrc_scores = _evaluate_one_matrix(
        qrc_bundle.features,
        targets,
        splits,
        first_valid=qrc_bundle.first_valid,
    ).set_index("target")
    qelm_scores = _evaluate_one_matrix(
        qelm_bundle.features,
        targets,
        splits,
        first_valid=qelm_bundle.first_valid,
    ).set_index("target")
    linear_scores = _evaluate_one_matrix(
        linear_features,
        targets,
        splits,
        first_valid=qelm_bundle.first_valid,
    ).set_index("target")
    poly_scores = _evaluate_one_matrix(
        poly_features,
        targets,
        splits,
        first_valid=qelm_bundle.first_valid,
    ).set_index("target")

    rows: list[dict[str, Any]] = []
    for tau in taus:
        for degree in degrees:
            target = _target_name(tau, degree)
            cap_qrc = float(qrc_scores.loc[target, "capacity_test"])
            cap_qelm = float(qelm_scores.loc[target, "capacity_test"])
            rows.append(
                {
                    "gamma": float(gamma),
                    "input_seed": int(input_seed),
                    "coupling_seed": int(coupling_seed),
                    "cell_id": f"{int(input_seed)}:{int(coupling_seed)}",
                    "family_cell": f"tau{int(tau)}_p{int(degree)}",
                    "target": target,
                    "tau": int(tau),
                    "p": int(degree),
                    "capacity_qrc": cap_qrc,
                    "capacity_qelm": cap_qelm,
                    "capacity_linear_delay": float(
                        linear_scores.loc[target, "capacity_test"]
                    ),
                    "capacity_poly_delay": float(poly_scores.loc[target, "capacity_test"]),
                    "delta": cap_qrc - cap_qelm,
                    "alpha_qrc": float(qrc_scores.loc[target, "alpha"]),
                    "alpha_qelm": float(qelm_scores.loc[target, "alpha"]),
                    "alpha_linear_delay": float(linear_scores.loc[target, "alpha"]),
                    "alpha_poly_delay": float(poly_scores.loc[target, "alpha"]),
                }
            )

    denom = float(recommendation["t_hi"] - recommendation["t_lo"])
    t_position = (
        float((qelm_cfg.T_evolve - recommendation["t_lo"]) / denom)
        if denom > 0.0
        else float("nan")
    )
    metadata = {
        "input_seed": int(input_seed),
        "coupling_seed": int(coupling_seed),
        "gamma": float(gamma),
        "qrc_config": qrc_bundle.config,
        "qelm_config": qelm_bundle.config,
        "qelm_recommendation": recommendation,
        "qelm_T_evolve_window_position": t_position,
        "feature_shape": {
            "qrc": list(qrc_bundle.features.shape),
            "qelm": list(qelm_bundle.features.shape),
            "linear_delay": list(linear_features.shape),
            "poly_delay": list(poly_features.shape),
        },
        "split_rows": {
            "train": int(splits.train.size),
            "val": int(splits.val.size),
            "test": int(splits.test.size),
            "first": int(splits.all[0]),
            "last": int(splits.all[-1]),
        },
        "n_obs": int(qrc_bundle.n_obs),
        "n_multiplex": int(qrc_bundle.n_multiplex),
        "readout_dim": D,
    }
    return rows, metadata


def _ci_excludes_zero(lo: float, hi: float) -> bool:
    return bool(np.isfinite([lo, hi]).all() and (float(lo) > 0.0 or float(hi) < 0.0))


def _label_from_record(row: dict[str, Any], *, tie_band: float) -> str:
    q_value = float(row["q_value"]) if np.isfinite(row["q_value"]) else None
    direction_label = stats.label_cell(
        row,
        tie_band=float(tie_band),
        q=q_value,
        q_thresh=FDR_Q,
    )
    significant = (
        _ci_excludes_zero(float(row["ci_low"]), float(row["ci_high"]))
        and q_value is not None
        and q_value <= FDR_Q
    )
    return direction_label if significant and direction_label in {"QRC", "QELM"} else "tie"


def _fractions(labels: list[str]) -> dict[str, float]:
    n = len(labels)
    if n == 0:
        return {"frac_qrc": 0.0, "frac_qelm": 0.0, "frac_tie": 0.0}
    return {
        "frac_qrc": float(labels.count("QRC") / n),
        "frac_qelm": float(labels.count("QELM") / n),
        "frac_tie": float(labels.count("tie") / n),
    }


def _aggregate_cells(
    rows: pd.DataFrame,
    *,
    seed: int,
    n_boot: int,
    tie_band: float = PRIMARY_TIE_BAND,
) -> pd.DataFrame:
    aggregates: list[dict[str, Any]] = []
    group_cols = ["gamma", "family_cell", "tau", "p"]
    for pos, ((gamma, family_cell, tau, p), group) in enumerate(
        rows.groupby(group_cols, sort=True)
    ):
        delta_stat = stats.paired_delta_stats(
            group["capacity_qrc"].to_numpy(),
            group["capacity_qelm"].to_numpy(),
            seed=int(seed) + pos,
            n_boot=int(n_boot),
        )
        qelm_linear_stat = stats.paired_delta_stats(
            group["capacity_qelm"].to_numpy(),
            group["capacity_linear_delay"].to_numpy(),
            seed=int(seed) + 10_000 + pos,
            n_boot=int(n_boot),
        )
        aggregates.append(
            {
                "gamma": float(gamma),
                "family_cell": str(family_cell),
                "tau": int(tau),
                "p": int(p),
                **delta_stat,
                "ci_width": float(delta_stat["ci_high"] - delta_stat["ci_low"]),
                "qelm_minus_linear_mean": float(qelm_linear_stat["mean_delta"]),
                "qelm_minus_linear_ci_low": float(qelm_linear_stat["ci_low"]),
                "qelm_minus_linear_ci_high": float(qelm_linear_stat["ci_high"]),
            }
        )

    out = pd.DataFrame(aggregates)
    if out.empty:
        return out
    out["q_value"] = stats.benjamini_hochberg(out["p_wilcoxon"].to_numpy())
    out["direction_label"] = [
        stats.label_cell(
            row,
            tie_band=tie_band,
            q=float(q_value) if np.isfinite(q_value) else None,
            q_thresh=FDR_Q,
        )
        for row, q_value in zip(out.to_dict(orient="records"), out["q_value"], strict=True)
    ]
    out["significant"] = [
        _ci_excludes_zero(float(lo), float(hi))
        and np.isfinite(q)
        and float(q) <= FDR_Q
        for lo, hi, q in zip(out["ci_low"], out["ci_high"], out["q_value"], strict=True)
    ]
    out["label"] = [
        direction_label if significant and direction_label in {"QRC", "QELM"} else "tie"
        for direction_label, significant in zip(
            out["direction_label"],
            out["significant"],
            strict=True,
        )
    ]
    return out


def _sensitivity(aggregate: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for tie_band in TIE_BANDS:
        labels = [_label_from_record(row, tie_band=float(tie_band)) for row in aggregate.to_dict(orient="records")]
        out[f"{tie_band:.2f}"] = _fractions(labels)
    return out


def _tie_counts(
    aggregate: pd.DataFrame,
    *,
    tie_band: float = PRIMARY_TIE_BAND,
) -> dict[str, int]:
    ties = aggregate[aggregate["label"] == "tie"]
    if ties.empty:
        return {"ties_power_limited": 0, "ties_genuine": 0}

    delta = ties["mean_delta"].to_numpy(dtype=float)
    ci_low = ties["ci_low"].to_numpy(dtype=float)
    ci_high = ties["ci_high"].to_numpy(dtype=float)
    finite = np.isfinite(delta) & np.isfinite(ci_low) & np.isfinite(ci_high)
    point_non_tie = np.abs(delta) >= float(tie_band)
    ci_straddles_zero = (ci_low <= 0.0) & (ci_high >= 0.0)
    return {
        "ties_power_limited": int(np.sum(finite & point_non_tie & ci_straddles_zero)),
        "ties_genuine": int(np.sum(finite & (np.abs(delta) < float(tie_band)))),
    }


def _coords(aggregate: pd.DataFrame, label: str) -> list[dict[str, int]]:
    region = aggregate[aggregate["label"] == label]
    return [
        {"tau": int(row.tau), "p": int(row.p)}
        for row in region.sort_values(["tau", "p"]).itertuples(index=False)
    ]


def _boundary_warnings(aggregate: pd.DataFrame, *, d: int) -> dict[str, Any]:
    qelm_beyond_delay = aggregate[
        (aggregate["label"] == "QELM") & (aggregate["tau"] >= int(d))
    ]
    qrc_short_linear = aggregate[
        (aggregate["label"] == "QRC")
        & (aggregate["tau"] < int(d))
        & (aggregate["p"] <= 2)
    ]
    warnings: list[str] = []
    if not qelm_beyond_delay.empty:
        warnings.append("QELM-labeled cells appear at tau >= d")
    if not qrc_short_linear.empty:
        warnings.append("QRC-labeled cells appear at tau < d and p <= 2")
    return {
        "contradicts_intuition": bool(warnings),
        "warnings": warnings,
        "qelm_beyond_delay": _coords(qelm_beyond_delay, "QELM"),
        "qrc_short_low_p": _coords(qrc_short_linear, "QRC"),
    }


def _dominant_case(frac_qrc: float, frac_qelm: float, frac_tie: float) -> str:
    if frac_qrc > 0.0 and frac_qelm == 0.0:
        return "qrc_dominates"
    if frac_qelm > 0.0 and frac_qrc == 0.0:
        return "qelm_dominates"
    if frac_tie >= 0.5:
        return "mostly_ties"
    return "mixed_but_below_gate"


def _diagnose(
    *,
    aggregate: pd.DataFrame,
    frac_qrc: float,
    frac_qelm: float,
    frac_tie: float,
    d: int,
    tie_band: float,
    t_positions: list[float],
    taus: tuple[int, ...],
    degrees: tuple[int, ...],
) -> dict[str, Any]:
    if frac_qrc >= STAGE1_MIN_REGIME_FRACTION and frac_qelm >= STAGE1_MIN_REGIME_FRACTION:
        return {
            "status": "GO",
            "reason": "both regimes occupy at least the required fraction",
            "suggested_tau_grid": list(taus),
            "suggested_p_grid": list(degrees),
        }

    case = _dominant_case(frac_qrc, frac_qelm, frac_tie)
    qelm_beats_linear_anywhere = bool(
        (
            (aggregate["p"] >= 2)
            & (aggregate["qelm_minus_linear_ci_low"] > 0.0)
        ).any()
    )
    ci_width = aggregate["ci_width"].to_numpy(dtype=float)
    underpowered = bool(np.nanmedian(ci_width) > 2.0 * float(tie_band))
    base = {
        "status": "NO-GO",
        "case": case,
        "qelm_beats_linear_delay_anywhere_p_ge_2": qelm_beats_linear_anywhere,
        "qelm_T_evolve_window_position_min": float(np.nanmin(t_positions)),
        "qelm_T_evolve_window_position_max": float(np.nanmax(t_positions)),
        "median_ci_width_gt_2x_tie_band": underpowered,
    }

    if case == "qrc_dominates":
        base.update(
            {
                "likely_issue": "QELM under-powered at this pilot setting",
                "suggested_tau_grid": [0, 1, 2, 3, 4, 5, 8],
                "suggested_p_grid": [1, 2, 3, 5],
                "suggested_model_check": "try a larger QELM delay window d after checking T_evolve remains inside the recommendation window",
            }
        )
    elif case == "qelm_dominates":
        base.update(
            {
                "likely_issue": "task family may be too memory-shallow",
                "suggested_tau_grid": [0, 1, 2, 3, 5, 8, 12, 16],
                "suggested_p_grid": [1, 2, 3, 5],
                "suggested_model_check": "extend the tau range beyond the current QELM delay window",
            }
        )
    elif case == "mostly_ties":
        base.update(
            {
                "likely_issue": "capacity ceiling/floor saturation or insufficient seed power",
                "suggested_tau_grid": [0, 1, 2, 3, 5, 8],
                "suggested_p_grid": [1, 2, 3, 4, 5, 7],
                "suggested_model_check": "increase pilot input seeds before freeze if CI widths remain large relative to the tie band",
            }
        )
    else:
        base.update(
            {
                "likely_issue": "a regime may exist but does not clear the material-fraction gate",
                "suggested_tau_grid": [0, 1, 2, 3, 4, 5, 8, 12],
                "suggested_p_grid": [1, 2, 3, 5, 7],
                "suggested_model_check": "densify tau or p near the observed minority-region coordinates",
            }
        )
    return base


def _slice_report(
    aggregate: pd.DataFrame,
    *,
    gamma: float,
    d: int,
    t_positions: list[float],
    taus: tuple[int, ...],
    degrees: tuple[int, ...],
) -> dict[str, Any]:
    primary = aggregate[aggregate["gamma"] == float(gamma)].copy()
    labels = primary["label"].tolist()
    fractions = _fractions(labels)
    q_values = primary["q_value"].to_numpy(dtype=float)
    ci_width = primary["ci_width"].to_numpy(dtype=float)
    tie_counts = _tie_counts(primary, tie_band=PRIMARY_TIE_BAND)
    go = bool(
        fractions["frac_qrc"] >= STAGE1_MIN_REGIME_FRACTION
        and fractions["frac_qelm"] >= STAGE1_MIN_REGIME_FRACTION
    )
    diagnostics = _diagnose(
        aggregate=primary,
        frac_qrc=fractions["frac_qrc"],
        frac_qelm=fractions["frac_qelm"],
        frac_tie=fractions["frac_tie"],
        d=d,
        tie_band=PRIMARY_TIE_BAND,
        t_positions=t_positions,
        taus=taus,
        degrees=degrees,
    )
    return {
        "gamma": float(gamma),
        "go": go,
        "fractions": fractions,
        **tie_counts,
        "region_coordinates": {
            "QRC": _coords(primary, "QRC"),
            "QELM": _coords(primary, "QELM"),
            "tie": _coords(primary, "tie"),
        },
        "tie_band_sensitivity": _sensitivity(primary),
        "noise_power": {
            "median_ci_width": float(np.nanmedian(ci_width)),
            "max_ci_width": float(np.nanmax(ci_width)),
            "n_cells_q_le_0_10": int(np.sum(np.isfinite(q_values) & (q_values <= FDR_Q))),
            "n_cells_ci_width_gt_tie_band": int(
                np.sum(np.isfinite(ci_width) & (ci_width > PRIMARY_TIE_BAND))
            ),
            "n_cells_ci_width_gt_2x_tie_band": int(
                np.sum(np.isfinite(ci_width) & (ci_width > 2.0 * PRIMARY_TIE_BAND))
            ),
            "n_cells": int(len(primary)),
            "tie_band": PRIMARY_TIE_BAND,
        },
        "boundary_check": _boundary_warnings(primary, d=d),
        "diagnostics": diagnostics,
    }


def run_design_pilot(
    profile_name: str,
    out_dir: Path,
    *,
    gammas: tuple[float, ...] | None = None,
    taus: tuple[int, ...] | None = None,
    degrees: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    base_profile = PROFILES[profile_name]
    gamma_slices = tuple(float(gamma) for gamma in (gammas or base_profile.gamma_slices))
    tau_axis = tuple(int(tau) for tau in (taus or TAUS))
    degree_axis = tuple(int(degree) for degree in (degrees or DEGREES))
    profile = replace(base_profile, gamma_slices=gamma_slices)
    all_rows: list[dict[str, Any]] = []
    seed_metadata: list[dict[str, Any]] = []
    for gamma in profile.gamma_slices:
        for cell in SEED_GRIDS["pilot"]:
            rows, metadata = _score_seed_cell(
                input_seed=cell.input_seed,
                coupling_seed=cell.coupling_seed,
                gamma=gamma,
                profile=profile,
                taus=tau_axis,
                degrees=degree_axis,
            )
            all_rows.extend(rows)
            seed_metadata.append(metadata)

    row_frame = pd.DataFrame(all_rows)
    aggregate = _aggregate_cells(
        row_frame,
        seed=BOOTSTRAP_SEED,
        n_boot=profile.n_boot,
        tie_band=PRIMARY_TIE_BAND,
    )

    d = int(seed_metadata[0]["qelm_config"]["d"])
    t_positions_by_gamma = {
        str(float(gamma)): [
            float(meta["qelm_T_evolve_window_position"])
            for meta in seed_metadata
            if float(meta["gamma"]) == float(gamma)
        ]
        for gamma in profile.gamma_slices
    }
    slice_reports = {
        str(float(gamma)): _slice_report(
            aggregate,
            gamma=float(gamma),
            d=d,
            t_positions=t_positions_by_gamma[str(float(gamma))],
            taus=tau_axis,
            degrees=degree_axis,
        )
        for gamma in profile.gamma_slices
    }
    primary_gamma = float(profile.gamma_slices[0])
    primary_report = slice_reports[str(primary_gamma)]
    passing_gammas = [
        float(gamma)
        for gamma in profile.gamma_slices
        if slice_reports[str(float(gamma))]["go"]
    ]

    result = {
        "profile": profile_name,
        "profile_config": asdict(profile),
        "status": "GO" if passing_gammas else "NO-GO",
        "primary_gamma": primary_gamma,
        "primary_report": primary_report,
        "passing_gammas": passing_gammas,
        "slice_reports": slice_reports,
        "cells": aggregate.sort_values(["gamma", "tau", "p"]).to_dict(orient="records"),
        "per_seed_scores": row_frame.sort_values(
            ["gamma", "input_seed", "coupling_seed", "tau", "p"]
        ).to_dict(orient="records"),
        "seed_metadata": seed_metadata,
        "seed_grid": grid_manifest("pilot"),
        "config": {
            "taus": list(tau_axis),
            "degrees": list(degree_axis),
            "n_qubits": 5,
            "n_virtual": 1,
            "observables": "default weight-2 3-axis Pauli observables",
            "alphas": list(ALPHAS),
            "tie_bands": list(TIE_BANDS),
            "fdr_q": FDR_Q,
            "min_regime_fraction": STAGE1_MIN_REGIME_FRACTION,
            "pilot_only": True,
            "confirmatory_use": "forbidden",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pilot.json"
    out_path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _format_coords(coords: list[dict[str, int]]) -> str:
    if not coords:
        return "none"
    return ", ".join(f"({coord['tau']},{coord['p']})" for coord in coords)


def _format_gammas(gammas: list[float]) -> str:
    if not gammas:
        return "none"
    return ", ".join(f"{float(gamma):.3g}" for gamma in gammas)


def _best_qrc_gamma(result: dict[str, Any]) -> float:
    gamma_order = [float(gamma) for gamma in result["profile_config"]["gamma_slices"]]
    candidates = result["passing_gammas"] or gamma_order
    return max(
        candidates,
        key=lambda gamma: (
            result["slice_reports"][str(float(gamma))]["fractions"]["frac_qrc"],
            result["slice_reports"][str(float(gamma))]["fractions"]["frac_qelm"],
            -gamma_order.index(gamma),
        ),
    )


def print_summary(result: dict[str, Any]) -> None:
    report = result["primary_report"]
    fractions = report["fractions"]
    power = report["noise_power"]
    print(f"design_pilot profile={result['profile']} status={result['status']}")
    print(f"passing_gammas: {_format_gammas(result['passing_gammas'])}")
    for gamma in result["profile_config"]["gamma_slices"]:
        slice_report = result["slice_reports"][str(float(gamma))]
        slice_fractions = slice_report["fractions"]
        print(
            "slice gamma={gamma:.3g} status={status} frac_qrc={frac_qrc:.3f} "
            "frac_qelm={frac_qelm:.3f} frac_tie={frac_tie:.3f} "
            "ties_power_limited={ties_power_limited} ties_genuine={ties_genuine}".format(
                gamma=float(gamma),
                status="GO" if slice_report["go"] else "NO-GO",
                ties_power_limited=slice_report["ties_power_limited"],
                ties_genuine=slice_report["ties_genuine"],
                **slice_fractions,
            )
        )
    best_gamma = _best_qrc_gamma(result)
    best_regions = result["slice_reports"][str(float(best_gamma))]["region_coordinates"]
    print(
        "best_qrc_gamma={gamma:.3g} QRC cells: {coords}".format(
            gamma=best_gamma,
            coords=_format_coords(best_regions["QRC"]),
        )
    )
    print(
        "primary gamma={gamma:.3g} frac_qrc={frac_qrc:.3f} "
        "frac_qelm={frac_qelm:.3f} frac_tie={frac_tie:.3f}".format(
            gamma=float(result["primary_gamma"]),
            **fractions,
        )
    )
    print(
        "noise_power median_ci_width={median:.4f} max_ci_width={maxw:.4f} "
        "q<=0.10={nq}/{n}".format(
            median=power["median_ci_width"],
            maxw=power["max_ci_width"],
            nq=power["n_cells_q_le_0_10"],
            n=power["n_cells"],
        )
    )
    for tie_band, sensitivity in report["tie_band_sensitivity"].items():
        print(
            "tie_band={band} frac_qrc={frac_qrc:.3f} "
            "frac_qelm={frac_qelm:.3f} frac_tie={frac_tie:.3f}".format(
                band=tie_band,
                **sensitivity,
            )
        )
    regions = report["region_coordinates"]
    print(f"QRC cells: {_format_coords(regions['QRC'])}")
    print(f"QELM cells: {_format_coords(regions['QELM'])}")
    print(f"tie cells: {_format_coords(regions['tie'])}")
    boundary = report["boundary_check"]
    if boundary["contradicts_intuition"]:
        print("boundary_warning: " + "; ".join(boundary["warnings"]))
    diagnostics = report["diagnostics"]
    if result["status"] != "GO":
        print(
            "no_go_diagnostic case={case} suggested_tau={taus} suggested_p={degrees}".format(
                case=diagnostics.get("case", "unknown"),
                taus=diagnostics.get("suggested_tau_grid"),
                degrees=diagnostics.get("suggested_p_grid"),
            )
        )
        print(f"no_go_check {diagnostics.get('suggested_model_check')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument(
        "--gammas",
        type=_parse_float_csv,
        default=None,
        help="comma-separated gamma slices; default is the selected profile",
    )
    parser.add_argument(
        "--taus",
        type=_parse_int_csv,
        default=None,
        help="comma-separated integer delay axis; default is the built-in TAUS",
    )
    parser.add_argument(
        "--degrees",
        type=_parse_int_csv,
        default=None,
        help="comma-separated integer degree axis; default is the built-in DEGREES",
    )
    args = parser.parse_args(argv)

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    result = run_design_pilot(
        args.profile,
        out_dir,
        gammas=args.gammas,
        taus=args.taus,
        degrees=args.degrees,
    )
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
