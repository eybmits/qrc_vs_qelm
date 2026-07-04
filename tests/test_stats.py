import numpy as np
import pandas as pd
import pytest

from qrc_vs_qelm import stats
from qrc_vs_qelm.stats import (
    add_q_values,
    benjamini_hochberg,
    label_cell,
    paired_delta_stats,
)


def test_paired_delta_stats_capacity_orientation_is_positive_for_qrc_win():
    assert "delta = capacity_QRC - capacity_QELM" in (stats.__doc__ or "")

    cap_qelm = np.array([0.40, 0.42, 0.45, 0.48, 0.44, 0.46, 0.43, 0.41])
    cap_qrc = cap_qelm + 0.08

    out = paired_delta_stats(cap_qrc, cap_qelm, seed=7, n_boot=256)

    assert out["mean_delta"] > 0.0
    assert out["delta_mean"] == pytest.approx(out["mean_delta"])
    assert out["win_rate"] > 0.5
    assert out["ci_low"] > 0.0
    assert out["delta_ci_low"] == pytest.approx(out["ci_low"])
    assert out["p_wilcoxon"] == pytest.approx(out["wilcoxon_p"], nan_ok=True)


def test_paired_delta_stats_degenerate_delta_reports_infinite_dz():
    cap_qelm = np.full(8, 0.50)
    cap_qrc = cap_qelm + 0.05

    out = paired_delta_stats(cap_qrc, cap_qelm, seed=17, n_boot=128)

    assert np.isinf(out["cohen_dz"])
    assert out["cohen_dz"] > 0.0


def test_benjamini_hochberg_known_vector_and_monotonicity():
    pvals = np.array([0.01, 0.04, 0.03, 0.002, np.nan])
    qvals = benjamini_hochberg(pvals)

    assert np.allclose(qvals[:4], np.array([0.02, 0.04, 0.04, 0.008]))
    assert np.isnan(qvals[4])

    valid = np.isfinite(qvals)
    order = np.argsort(pvals[valid])
    assert np.all(np.diff(qvals[valid][order]) >= -1e-12)


def test_add_q_values_returns_copy():
    df = pd.DataFrame({"p": [0.01, 0.02, 0.5]})
    out = add_q_values(df, "p", "q")

    assert "q" in out
    assert "q" not in df
    assert out.loc[0, "q"] <= out.loc[2, "q"]


def test_label_cell_tie_band_signs_and_q_threshold():
    assert label_cell({"mean_delta": 0.01}) == "tie"
    assert label_cell({"mean_delta": 0.03}) == "QRC"
    assert label_cell({"mean_delta": -0.03}) == "QELM"
    assert label_cell({"mean_delta": 0.20}, q=0.11) == "tie"
    assert label_cell(-0.20, q=0.01) == "QELM"
