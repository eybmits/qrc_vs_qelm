from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analyze_n4_shots import analyze_n4_shots


def test_n4_shot_profile_analyzes_and_rejects_off_grid_budget() -> None:
    repo = Path(__file__).resolve().parents[2]
    run_root = (
        repo
        / "pra_extension"
        / "outputs"
        / "finite_shots_n4_exploratory_v1_12roots"
    )
    results = pd.read_csv(run_root / "finite_shot_results.csv")
    roots, summary = analyze_n4_shots(results)
    assert len(roots) == 48
    assert summary["counts"] == {"rows": 324, "roots": 12, "teacher_sizes": [4]}

    mutated = results.copy()
    finite_index = mutated["allocation_mode"].eq("equal").idxmax()
    mutated.loc[finite_index, "total_shots"] = 301
    with pytest.raises(ValueError, match="off grid"):
        analyze_n4_shots(mutated)
