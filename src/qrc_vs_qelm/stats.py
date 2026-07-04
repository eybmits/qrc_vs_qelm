"""Statistics for capacity comparisons.

The signed convention throughout this module is
``delta = capacity_QRC - capacity_QELM``.  Capacity is higher-is-better, so a
positive delta means QRC wins and a negative delta means QELM wins.  Gates,
confidence intervals, labels, and win rates derive from this signed delta.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


DZ_STD_EPS = 1.0e-12


def bootstrap_ci(
    values: Iterable[float],
    seed: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""

    arr = np.asarray(list(values), dtype=float)
    if len(arr) <= 1:
        mean = float(arr.mean()) if len(arr) else float("nan")
        return mean, mean

    n_boot = int(n_boot)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n = len(arr)
    for i in range(n_boot):
        boot[i] = float(arr[rng.integers(0, n, n)].mean())

    lo, hi = np.percentile(boot, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def _wilcoxon_delta_p(delta: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon

        return float(
            wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue
        )
    except Exception:
        return float("nan")


def paired_delta_stats(
    cap_qrc: Iterable[float],
    cap_qelm: Iterable[float],
    *,
    seed: int,
    n_boot: int = 2000,
) -> dict[str, float | int]:
    """Return paired capacity-delta statistics over seed cells.

    Positive ``mean_delta`` means QRC has higher capacity than QELM.
    """

    qrc = np.asarray(list(cap_qrc), dtype=float)
    qelm = np.asarray(list(cap_qelm), dtype=float)
    if qrc.shape != qelm.shape:
        raise ValueError(f"cap_qrc/cap_qelm shape mismatch: {qrc.shape} vs {qelm.shape}")
    if len(qrc) == 0:
        raise ValueError("paired_delta_stats requires at least one pair")
    if not np.isfinite(qrc).all() or not np.isfinite(qelm).all():
        raise ValueError("capacities must be finite")

    delta = qrc - qelm
    ci_low, ci_high = bootstrap_ci(delta, seed=seed, n_boot=n_boot)
    delta_std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
    delta_se = delta_std / float(np.sqrt(len(delta))) if len(delta) > 1 else 0.0
    mean_delta = float(delta.mean())
    if len(delta) <= 1:
        dz = float("nan")
    elif delta_std <= DZ_STD_EPS:
        dz = (
            float(np.copysign(np.inf, mean_delta))
            if mean_delta != 0.0
            else float("nan")
        )
    else:
        dz = mean_delta / delta_std
    p_wilcoxon = _wilcoxon_delta_p(delta)

    return {
        "n_pairs": int(len(delta)),
        "capacity_qrc_mean": float(qrc.mean()),
        "capacity_qelm_mean": float(qelm.mean()),
        "delta_mean": mean_delta,
        "mean_delta": mean_delta,
        "delta_std": delta_std,
        "delta_se": delta_se,
        "se": delta_se,
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohen_dz": float(dz),
        "win_rate": float(np.mean(delta > 0.0)),
        "p_wilcoxon": p_wilcoxon,
        "wilcoxon_p": p_wilcoxon,
    }


def benjamini_hochberg(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    adjusted = ranked * m / np.arange(1, len(ranked) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out_valid = np.empty_like(pv)
    out_valid[order] = adjusted
    q[valid] = out_valid
    return q


def add_q_values(df: pd.DataFrame, p_col: str, q_col: str) -> pd.DataFrame:
    """Return a copy of ``df`` with BH-FDR q-values added."""

    out = df.copy()
    out[q_col] = benjamini_hochberg(out[p_col].to_numpy()) if len(out) else []
    return out


def _mean_delta(delta_stat: Mapping[str, Any] | float) -> float:
    if isinstance(delta_stat, Mapping):
        for key in ("mean_delta", "delta_mean", "delta"):
            if key in delta_stat:
                return float(delta_stat[key])
        raise KeyError("delta_stat must contain mean_delta, delta_mean, or delta")
    return float(delta_stat)


def label_cell(
    delta_stat: Mapping[str, Any] | float,
    tie_band: float = 0.02,
    q: float | None = None,
    q_thresh: float = 0.10,
) -> str:
    """Label one family cell as QRC, QELM, or tie from the signed delta."""

    mean_delta = _mean_delta(delta_stat)
    if abs(mean_delta) < float(tie_band):
        return "tie"
    if q is not None and float(q) > float(q_thresh):
        return "tie"
    return "QRC" if mean_delta > 0.0 else "QELM"


__all__ = [
    "bootstrap_ci",
    "DZ_STD_EPS",
    "paired_delta_stats",
    "benjamini_hochberg",
    "add_q_values",
    "label_cell",
]
