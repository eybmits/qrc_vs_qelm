"""Frozen gates and verdict evaluation for pre-registered score tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import re
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from qrc_vs_qelm.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    label_cell,
    paired_delta_stats,
)


FDR_Q = 0.10
TIE_BAND = 0.02
STAGE1_MIN_REGIME_FRACTION = 0.20
QELM_LINEAR_DELAY_FLOOR = 0.25
BOOTSTRAP_SEED = 20260704


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


_GATE_DEFS: dict[str, dict[str, Any]] = {
    "G1a": {
        "stage": 1,
        "description": "Per-family-cell paired capacity delta has a CI excluding zero with BH-FDR q <= 0.10.",
        "kind": "per_cell_significance",
        "threshold": 0.0,
        "kill": False,
        "required": False,
        "failure_verdict": "TIE_OR_NON_SIGNIFICANT",
    },
    "G1b": {
        "stage": 1,
        "description": "Both significant QRC-favored and significant QELM-favored family-cell regimes occupy at least 20% of cells.",
        "kind": "headline_two_regime",
        "threshold": {
            "min_qrc_fraction": STAGE1_MIN_REGIME_FRACTION,
            "min_qelm_fraction": STAGE1_MIN_REGIME_FRACTION,
        },
        "kill": False,
        "required": True,
        "failure_verdict": "SINGLE_REGIME",
    },
    "G1c": {
        "stage": 1,
        "description": "QELM beats the linear-delay control on at least 25% of p>=2 family cells.",
        "kind": "kill_qelm_trivial",
        "threshold": QELM_LINEAR_DELAY_FLOOR,
        "kill": True,
        "required": True,
        "failure_verdict": "QELM_TRIVIAL",
    },
    "G2a": {
        "stage": 2,
        "description": "The QRC-favored cell count changes between S=infinity and S=1e4 with a bootstrap CI excluding zero.",
        "kind": "shot_sensitivity",
        "threshold": 0.0,
        "kill": False,
        "required": True,
        "failure_verdict": "SHOT_INSENSITIVE",
    },
    "G3a": {
        "stage": 3,
        "description": "MC(gamma) is non-increasing, NC(gamma) is non-decreasing, and a QELM>=QRC crossover exists.",
        "kind": "gamma_tradeoff",
        "threshold": 0.0,
        "kill": False,
        "required": True,
        "failure_verdict": "NO_TRADEOFF",
    },
}


GATE_DEFS: Mapping[str, Mapping[str, Any]] = _deep_freeze(_GATE_DEFS)


@dataclass(frozen=True)
class GateRecord:
    gate_id: str
    passed: bool
    verdict: str
    value: Any
    threshold: Any
    ci_low: Any = None
    ci_high: Any = None
    q_value: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""

        return asdict(self)


def evaluate_stage1_gates(
    df: pd.DataFrame,
    *,
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = 2000,
) -> list[GateRecord]:
    """Evaluate Stage 1 per-cell, headline, and linear-delay gates."""

    scores = _with_delta(df)
    cell_stats = _family_cell_delta_stats(scores, seed=seed, n_boot=n_boot)
    q_values = benjamini_hochberg([stat["p_wilcoxon"] for stat in cell_stats])

    records: list[GateRecord] = []
    labels: list[str] = []
    gate_labels: list[str] = []
    for stat, q_value in zip(cell_stats, q_values, strict=True):
        passed = (
            _ci_excludes_zero(stat["ci_low"], stat["ci_high"])
            and np.isfinite(q_value)
            and float(q_value) <= FDR_Q
        )
        label = label_cell(
            stat,
            tie_band=TIE_BAND,
            q=float(q_value) if np.isfinite(q_value) else None,
            q_thresh=FDR_Q,
        )
        labels.append(label)
        gate_labels.append(label if passed and label in {"QRC", "QELM"} else "tie")
        records.append(
            GateRecord(
                gate_id="G1a",
                passed=passed,
                verdict="PASS" if passed else "TIE_OR_NON_SIGNIFICANT",
                value=stat["mean_delta"],
                threshold=GATE_DEFS["G1a"]["threshold"],
                ci_low=stat["ci_low"],
                ci_high=stat["ci_high"],
                q_value=float(q_value) if np.isfinite(q_value) else float("nan"),
                notes=f"family_cell={stat['family_cell']} label={label}",
            )
        )

    records.append(_evaluate_g1b(gate_labels, direction_labels=labels))
    records.append(_evaluate_g1c(scores, seed=seed, n_boot=n_boot))
    return records


def evaluate_stage2_gates(
    df: pd.DataFrame,
    *,
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = 2000,
) -> list[GateRecord]:
    """Evaluate the Stage 2 shot-sensitivity gate."""

    scores = _with_delta(df)
    shot_col = _first_existing_column(scores, ("shots", "shot_count", "S", "n_shots"))
    if shot_col is None:
        return [
            GateRecord(
                gate_id="G2a",
                passed=False,
                verdict="SHOT_INSENSITIVE",
                value=float("nan"),
                threshold=GATE_DEFS["G2a"]["threshold"],
                notes="missing shot-count column",
            )
        ]

    inf_scores = scores[_is_infinite_shots(scores[shot_col])]
    finite_scores = scores[_is_shot_1e4(scores[shot_col])]
    if inf_scores.empty or finite_scores.empty:
        return [
            GateRecord(
                gate_id="G2a",
                passed=False,
                verdict="SHOT_INSENSITIVE",
                value=float("nan"),
                threshold=GATE_DEFS["G2a"]["threshold"],
                notes="requires both S=infinity and S=1e4 rows",
            )
        ]

    counts_inf = _qrc_favored_counts(inf_scores)
    counts_1e4 = _qrc_favored_counts(finite_scores)
    diffs = _paired_count_differences(counts_inf, counts_1e4)
    if len(diffs) == 0:
        return [
            GateRecord(
                gate_id="G2a",
                passed=False,
                verdict="SHOT_INSENSITIVE",
                value=float("nan"),
                threshold=GATE_DEFS["G2a"]["threshold"],
                notes="no paired count strata between shot tiers",
            )
        ]

    ci_low, ci_high = bootstrap_ci(diffs, seed=seed, n_boot=n_boot)
    mean_diff = float(np.mean(diffs))
    passed = _ci_excludes_zero(ci_low, ci_high)
    return [
        GateRecord(
            gate_id="G2a",
            passed=passed,
            verdict="PASS" if passed else "SHOT_INSENSITIVE",
            value=mean_diff,
            threshold=GATE_DEFS["G2a"]["threshold"],
            ci_low=ci_low,
            ci_high=ci_high,
            notes="mean count difference is S=1e4 minus S=infinity",
        )
    ]


def evaluate_stage3_gates(
    df: pd.DataFrame,
    *,
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = 2000,
) -> list[GateRecord]:
    """Evaluate the Stage 3 gamma tradeoff gate."""

    scores = _with_delta(df)
    gamma_col = _first_existing_column(scores, ("gamma", "dissipation", "gamma_value"))
    if gamma_col is None:
        return [
            GateRecord(
                gate_id="G3a",
                passed=False,
                verdict="NO_TRADEOFF",
                value=float("nan"),
                threshold=GATE_DEFS["G3a"]["threshold"],
                notes="missing gamma column",
            )
        ]

    trend = _gamma_trend_frame(scores, gamma_col)
    if trend.empty or trend["mc"].notna().sum() < 2 or trend["nc"].notna().sum() < 2:
        return [
            GateRecord(
                gate_id="G3a",
                passed=False,
                verdict="NO_TRADEOFF",
                value=float("nan"),
                threshold=GATE_DEFS["G3a"]["threshold"],
                notes="insufficient MC/NC trend data",
            )
        ]

    mc_rho, mc_lo, mc_hi = _spearman_with_ci(
        trend[gamma_col].to_numpy(),
        trend["mc"].to_numpy(),
        seed=seed,
        n_boot=n_boot,
    )
    nc_rho, nc_lo, nc_hi = _spearman_with_ci(
        trend[gamma_col].to_numpy(),
        trend["nc"].to_numpy(),
        seed=seed + 31,
        n_boot=n_boot,
    )
    gamma_star, crossover_fraction = _gamma_crossover(scores, gamma_col)
    passed = (
        np.isfinite(mc_hi)
        and np.isfinite(nc_lo)
        and mc_hi <= 0.0
        and nc_lo >= 0.0
        and gamma_star is not None
    )
    value = {
        "mc_rho": mc_rho,
        "nc_rho": nc_rho,
        "gamma_star": gamma_star,
        "crossover_fraction": crossover_fraction,
    }
    return [
        GateRecord(
            gate_id="G3a",
            passed=passed,
            verdict="PASS" if passed else "NO_TRADEOFF",
            value=value,
            threshold=GATE_DEFS["G3a"]["threshold"],
            ci_low={"mc": mc_lo, "nc": nc_lo},
            ci_high={"mc": mc_hi, "nc": nc_hi},
            notes="Spearman CI uses bootstrap over gamma aggregate points",
        )
    ]


def stage_verdict(records: Iterable[GateRecord], stage: int) -> dict:
    """Combine gate records into one stage verdict."""

    stage_gate_ids = {
        gate_id
        for gate_id, gate in GATE_DEFS.items()
        if int(gate.get("stage", -1)) == int(stage)
    }
    stage_records = [record for record in records if record.gate_id in stage_gate_ids]
    failed_kills = [
        record
        for record in stage_records
        if GATE_DEFS[record.gate_id]["kill"] and not record.passed
    ]
    if failed_kills:
        first = failed_kills[0]
        return {
            "stage": int(stage),
            "passed": False,
            "verdict": first.verdict,
            "failed_gates": [record.gate_id for record in failed_kills],
        }

    required_ids = {
        gate_id for gate_id in stage_gate_ids if GATE_DEFS[gate_id].get("required", True)
    }
    missing_required = sorted(
        gate_id
        for gate_id in required_ids
        if not any(record.gate_id == gate_id for record in stage_records)
    )
    if missing_required:
        return {
            "stage": int(stage),
            "passed": False,
            "verdict": "MISSING_GATES",
            "failed_gates": missing_required,
        }

    failed_required = [
        record
        for record in stage_records
        if record.gate_id in required_ids and not record.passed
    ]
    if failed_required:
        first = failed_required[0]
        return {
            "stage": int(stage),
            "passed": False,
            "verdict": first.verdict,
            "failed_gates": [record.gate_id for record in failed_required],
        }

    return {
        "stage": int(stage),
        "passed": True,
        "verdict": "PASS",
        "failed_gates": [],
    }


def gates_markdown_table() -> str:
    """Render the frozen gate definitions as a markdown table."""

    lines = [
        "| Gate | Stage | Kind | Description | Threshold | Kill | Required |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for gate_id, gate in GATE_DEFS.items():
        lines.append(
            "| {gate_id} | {stage} | {kind} | {description} | {threshold} | {kill} | {required} |".format(
                gate_id=_md(gate_id),
                stage=gate["stage"],
                kind=_md(str(gate["kind"])),
                description=_md(str(gate["description"])),
                threshold=_md(str(gate["threshold"])),
                kill=str(bool(gate["kill"])),
                required=str(bool(gate.get("required", True))),
            )
        )
    return "\n".join(lines)


def _with_delta(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"capacity_qrc", "capacity_qelm"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"missing required score columns: {missing}")
    if "delta" not in out.columns:
        out["delta"] = out["capacity_qrc"].to_numpy(dtype=float) - out[
            "capacity_qelm"
        ].to_numpy(dtype=float)
    return out


def _family_col(df: pd.DataFrame) -> str:
    if "family_cell" in df.columns:
        return "family_cell"
    if "target" in df.columns:
        return "target"
    raise ValueError("scores must include family_cell or target")


def _family_cell_delta_stats(
    scores: pd.DataFrame,
    *,
    seed: int,
    n_boot: int,
) -> list[dict[str, Any]]:
    family_col = _family_col(scores)
    out: list[dict[str, Any]] = []
    for pos, (family_cell, group) in enumerate(scores.groupby(family_col, sort=False)):
        stat = paired_delta_stats(
            group["capacity_qrc"].to_numpy(),
            group["capacity_qelm"].to_numpy(),
            seed=seed + pos,
            n_boot=n_boot,
        )
        stat["family_cell"] = family_cell
        out.append(stat)
    return out


def _evaluate_g1b(
    labels: list[str],
    *,
    direction_labels: list[str] | None = None,
) -> GateRecord:
    """Evaluate G1b using only cells that also pass the G1a significance rule.

    ``direction_labels`` preserves the direction/tie labels useful for plots; the
    gate labels passed as ``labels`` count non-significant cells as ties.
    """

    n = len(labels)
    qrc_fraction = float(labels.count("QRC") / n) if n else 0.0
    qelm_fraction = float(labels.count("QELM") / n) if n else 0.0
    tie_fraction = float(labels.count("tie") / n) if n else 0.0
    direction_counts = direction_labels or labels
    passed = (
        qrc_fraction >= STAGE1_MIN_REGIME_FRACTION
        and qelm_fraction >= STAGE1_MIN_REGIME_FRACTION
    )
    return GateRecord(
        gate_id="G1b",
        passed=passed,
        verdict="PASS" if passed else "SINGLE_REGIME",
        value={
            "qrc_fraction": qrc_fraction,
            "qelm_fraction": qelm_fraction,
            "tie_fraction": tie_fraction,
            "n_cells": n,
        },
        threshold=GATE_DEFS["G1b"]["threshold"],
        notes=(
            "gate counts significant cells only; "
            f"significant_labels: QRC={labels.count('QRC')} "
            f"QELM={labels.count('QELM')} tie={labels.count('tie')}; "
            f"direction_labels: QRC={direction_counts.count('QRC')} "
            f"QELM={direction_counts.count('QELM')} "
            f"tie={direction_counts.count('tie')}"
        ),
    )


def _evaluate_g1c(
    scores: pd.DataFrame,
    *,
    seed: int,
    n_boot: int,
) -> GateRecord:
    linear_col = _first_existing_column(
        scores,
        (
            "capacity_linear_delay",
            "linear_delay_capacity",
            "linear_delay",
            "capacity_delay",
        ),
    )
    if linear_col is None:
        return GateRecord(
            gate_id="G1c",
            passed=False,
            verdict="QELM_TRIVIAL",
            value=float("nan"),
            threshold=GATE_DEFS["G1c"]["threshold"],
            notes="missing linear-delay control capacity",
        )

    work = scores.copy()
    work["_degree"] = _degree_series(work)
    work = work[work["_degree"] >= 2]
    if work.empty:
        fraction = 0.0
    else:
        family_col = _family_col(work)
        beats = []
        for pos, (_, group) in enumerate(work.groupby(family_col, sort=False)):
            stat = paired_delta_stats(
                group["capacity_qelm"].to_numpy(),
                group[linear_col].to_numpy(),
                seed=seed + 10_000 + pos,
                n_boot=n_boot,
            )
            beats.append(float(stat["ci_low"]) > 0.0)
        fraction = float(np.mean(beats)) if beats else 0.0

    passed = fraction >= QELM_LINEAR_DELAY_FLOOR
    return GateRecord(
        gate_id="G1c",
        passed=passed,
        verdict="PASS" if passed else "QELM_TRIVIAL",
        value=fraction,
        threshold=GATE_DEFS["G1c"]["threshold"],
        notes="fraction of p>=2 family cells where QELM-minus-linear-delay bootstrap CI is positive",
    )


def _degree_series(df: pd.DataFrame) -> pd.Series:
    if "p" in df.columns:
        return pd.to_numeric(df["p"], errors="coerce").fillna(-np.inf)
    if "degree" in df.columns:
        return pd.to_numeric(df["degree"], errors="coerce").fillna(-np.inf)
    family_col = _family_col(df)
    return df[family_col].map(_degree_from_family_cell).astype(float).fillna(-np.inf)


def _degree_from_family_cell(value: Any) -> float:
    if isinstance(value, Mapping):
        for key in ("p", "degree"):
            if key in value:
                return float(value[key])
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return float(value[1])
    text = str(value)
    patterns = (
        r"(?:^|[_\s,\(])p(?:=|_)?(\d+)",
        r"degree(?:=|_)?(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return float("nan")


def _ci_excludes_zero(lo: Any, hi: Any) -> bool:
    try:
        lo_f = float(lo)
        hi_f = float(hi)
    except (TypeError, ValueError):
        return False
    return np.isfinite(lo_f) and np.isfinite(hi_f) and (lo_f > 0.0 or hi_f < 0.0)


def _first_existing_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _is_infinite_shots(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.lower()
    return numeric.eq(np.inf) | text.isin({"inf", "infinity", "s=inf", "s=infinity"})


def _is_shot_1e4(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.lower()
    return np.isclose(numeric, 1.0e4) | text.isin({"1e4", "10000", "s=1e4", "s=10000"})


def _qrc_favored_counts(df: pd.DataFrame) -> dict[Any, int]:
    family_col = _family_col(df)
    strata_cols = [
        col
        for col in ("cell_id", "input_seed", "coupling_seed", "shot_seed")
        if col in df.columns
    ]
    if not strata_cols:
        strata_cols = ["__all__"]
        df = df.assign(__all__="all")

    counts: dict[Any, int] = {}
    group_key: str | list[str] = strata_cols[0] if len(strata_cols) == 1 else strata_cols
    for key, stratum in df.groupby(group_key, sort=False):
        favored = []
        for _, family in stratum.groupby(family_col, sort=False):
            favored.append(float(family["delta"].mean()) > 0.0)
        counts[key] = int(np.sum(favored))
    return counts


def _paired_count_differences(
    counts_inf: Mapping[Any, int],
    counts_1e4: Mapping[Any, int],
) -> np.ndarray:
    keys = [key for key in counts_inf if key in counts_1e4]
    return np.asarray([counts_1e4[key] - counts_inf[key] for key in keys], dtype=float)


def _gamma_trend_frame(scores: pd.DataFrame, gamma_col: str) -> pd.DataFrame:
    explicit_mc = _first_existing_column(scores, ("mc", "MC", "memory_capacity"))
    explicit_nc = _first_existing_column(scores, ("nc", "NC", "nonlinear_capacity"))
    rows: list[dict[str, float]] = []
    if explicit_mc is not None and explicit_nc is not None:
        for gamma, group in scores.groupby(gamma_col, sort=True):
            rows.append(
                {
                    gamma_col: float(gamma),
                    "mc": float(group[explicit_mc].mean()),
                    "nc": float(group[explicit_nc].mean()),
                }
            )
        return pd.DataFrame(rows)

    work = scores.copy()
    work["_degree"] = _degree_series(work)
    for gamma, group in work.groupby(gamma_col, sort=True):
        mc = group.loc[group["_degree"] <= 1, "capacity_qrc"]
        nc = group.loc[group["_degree"] >= 2, "capacity_qrc"]
        rows.append(
            {
                gamma_col: float(gamma),
                "mc": float(mc.mean()) if len(mc) else float("nan"),
                "nc": float(nc.mean()) if len(nc) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_boot: int,
) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")

    rho = _spearman_rho(x, y)
    if len(x) <= 2:
        return rho, rho, rho

    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(x), len(x))
        value = _spearman_rho(x[idx], y[idx])
        if np.isfinite(value):
            boot.append(value)
    if not boot:
        return rho, float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(boot, dtype=float), [2.5, 97.5])
    return rho, float(lo), float(hi)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y).statistic)
    except Exception:
        return float("nan")


def _gamma_crossover(scores: pd.DataFrame, gamma_col: str) -> tuple[float | None, float]:
    family_col = _family_col(scores)
    best_gamma: float | None = None
    best_fraction = 0.0
    for gamma, group in scores.groupby(gamma_col, sort=True):
        outcomes = []
        for _, family in group.groupby(family_col, sort=False):
            outcomes.append(float(family["capacity_qelm"].mean()) >= float(family["capacity_qrc"].mean()))
        fraction = float(np.mean(outcomes)) if outcomes else 0.0
        if fraction > best_fraction:
            best_fraction = fraction
        if fraction >= 0.5 and best_gamma is None:
            best_gamma = float(gamma)
    return best_gamma, best_fraction


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


__all__ = [
    "GateRecord",
    "GATE_DEFS",
    "evaluate_stage1_gates",
    "evaluate_stage2_gates",
    "evaluate_stage3_gates",
    "stage_verdict",
    "gates_markdown_table",
]
