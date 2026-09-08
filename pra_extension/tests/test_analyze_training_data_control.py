"""Regression checks for paired inference without inflating sample size."""
import numpy as np
import pandas as pd
import pytest

from analyze_training_data_control import holm_adjust, paired_summary, render


def test_positive_paired_reduction_and_root_denominator():
    result = paired_summary(np.ones(12), np.full(12, .5), seed=7, replicates=1000)
    assert result["roots"] == result["positive_roots"] == 12
    assert result["median_decrease"] == .5
    assert result["decrease_ci95"] == [.5, .5]
    assert result["one_sided_sign_p"] == 1 / 4096


def test_negative_changes_and_ties_are_not_discarded():
    result = paired_summary(np.ones(12), np.r_[np.ones(4), np.full(8, 2.)], seed=7, replicates=100)
    assert result["positive_roots"] == 0
    assert result["median_decrease"] == -1
    assert result["one_sided_sign_p"] == 1


def test_zero_baseline_retains_absolute_inference():
    result = paired_summary(np.zeros(12), np.zeros(12), seed=7, replicates=100)
    assert result["median_relative_decrease"] is None
    assert result["median_decrease"] == 0


def test_holm_keeps_original_order_and_caps():
    assert holm_adjust([.03, .01]) == [.03, .02]
    assert holm_adjust([.8, .7]) == [1., 1.]


@pytest.mark.parametrize("low,high", [([1], [1, 2]), ([], []), ([np.nan], [1]), ([-1], [1])])
def test_invalid_pairs_fail(low, high):
    with pytest.raises(ValueError):
        paired_summary(np.array(low), np.array(high), seed=1, replicates=10)


def test_static_figure_exports_both_formats(tmp_path):
    # Synthetic values exercise layout only; they never enter study evidence.
    anchors = [{"id": "n4_fast", "teacher_n_qubits": 4, "window": 6, "gamma": 4},
               {"id": "n5_long", "teacher_n_qubits": 5, "window": 10, "gamma": 1}]
    counts = [480, 960, 1920, 3840]
    rows = []
    for anchor in anchors:
        for model, budget in [("ptd_parallel", 1), ("ptd_parallel", 2),
                              ("ptd_parallel", 4), ("poly2_delay", 0)]:
            for count in counts:
                for root in range(12):
                    rows.append({"anchor_id": anchor["id"], "model": model,
                                 "snapshot_budget": budget, "training_rows": count,
                                 "reference_mean_nrmse": (.04 + .02 / (budget + 1))
                                 * (480 / count)**.15 * (1 + root / 50)})
    render(pd.DataFrame(rows), {"anchors": anchors, "training_rows": counts}, tmp_path)
    assert (tmp_path / "training_data_control.pdf").stat().st_size > 1000
    assert (tmp_path / "training_data_control.png").stat().st_size > 1000
