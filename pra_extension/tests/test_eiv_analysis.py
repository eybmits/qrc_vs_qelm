from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analyze_eiv_shots import analyze_eiv


def _results() -> pd.DataFrame:
    repo = Path(__file__).resolve().parents[2]
    return pd.read_csv(
        repo
        / "pra_extension"
        / "outputs"
        / "finite_shots_eiv_n4_exploratory_v2_final_helpers_12roots"
        / "eiv_finite_shot_results.csv"
    )


def test_eiv_strict_analysis_accepts_exact_profile_and_root_blocks() -> None:
    roots, selected, summary = analyze_eiv(_results())
    assert len(roots) == 12 * 4
    assert len(selected) == 12 * 4 * 16
    assert summary["counts"] == {
        "candidate_rows": 2_304,
        "selected_test_rows": 768,
        "roots": 12,
        "replicates_per_root_resource_cell": 16,
    }
    assert summary["strict_invariants"]["unique_split_entropy_tuples"] == 6_912


def test_eiv_strict_analysis_rejects_test_driven_budget_flag() -> None:
    mutated = _results()
    group = mutated[
        mutated["root_id"].eq("eivn4_01")
        & mutated["total_shots"].eq(300)
        & mutated["replicate"].eq(0)
    ]
    selected_index = group[group["selected_budget_on_validation"]].index[0]
    replacement_index = group[
        ~group["selected_budget_on_validation"]
    ].sort_values("test_noisy_feature_nrmse").index[0]
    mutated.loc[selected_index, "selected_budget_on_validation"] = False
    mutated.loc[replacement_index, "selected_budget_on_validation"] = True
    with pytest.raises(ValueError, match="selected by validation only"):
        analyze_eiv(mutated)


def test_eiv_strict_analysis_rejects_reused_split_stream() -> None:
    mutated = _results()
    mutated.loc[1, "test_entropy"] = mutated.loc[0, "test_entropy"]
    with pytest.raises(ValueError, match="entropy"):
        analyze_eiv(mutated)
