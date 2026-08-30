from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from analyze_continuous_memory import analyze_continuous


def test_continuous_profile_analyzes_and_rejects_mutations() -> None:
    repo = Path(__file__).resolve().parents[2]
    run_root = (
        repo
        / "pra_extension"
        / "outputs"
        / "observable_memory_continuous_exploratory_v1_12roots"
    )
    results = pd.read_csv(run_root / "observable_memory_results.csv")
    run_summary = json.loads(
        (run_root / "observable_memory_summary.json").read_text(encoding="utf-8")
    )
    roots, summary = analyze_continuous(results, run_summary)
    assert len(roots) == 36
    assert summary["counts"] == {"rows": 648, "roots": 12}

    quantized = copy.deepcopy(run_summary)
    quantized["config"]["quantize_levels"] = 13
    with pytest.raises(ValueError, match="unsnapped continuous"):
        analyze_continuous(results, quantized)

    off_grid = results.copy()
    off_grid.loc[0, "reference_local_loss_gamma"] = 3.5
    with pytest.raises(ValueError, match="Cartesian profile"):
        analyze_continuous(off_grid, run_summary)
