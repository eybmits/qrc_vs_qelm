from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_dense_phase_secondary import aggregate, validate_results  # noqa: E402
from dense_phase_secondary_common import (  # noqa: E402
    CONFIG_PATH,
    assert_seed_disjointness,
    validate_dense_configuration,
    verify_dense_freeze,
)


RESULTS = ROOT / "results_dense_phase_secondary_v1"
ANALYSIS = ROOT / "results_dense_phase_secondary_v1_analysis"


def test_dense_secondary_material_was_frozen_and_seeds_are_disjoint() -> None:
    assert validate_dense_configuration(CONFIG_PATH) == {
        "passed": True,
        "history_root_cells": 1_584,
        "ptd_root_budget_cells": 4_752,
        "expected_science_rows_per_table": 15_840,
        "root_set_sha256": "4c8fac57780c9d4964aa1b9562b1f8bb94f503dc97154f3fc8b87e32d238a4f7",
    }
    seed_audit = assert_seed_disjointness(CONFIG_PATH)
    assert seed_audit["passed"] is True
    assert seed_audit["dense_seed_count"] == 108
    freeze = verify_dense_freeze(CONFIG_PATH)
    assert freeze["passed"] is True
    assert all(freeze["checks"].values())


def test_dense_secondary_passes_the_frozen_checks() -> None:
    validated = validate_results(RESULTS, CONFIG_PATH)
    cells, scaling, summary = aggregate(validated, CONFIG_PATH)
    assert summary["decision"] == "PASSES_DENSE_SECONDARY"
    assert summary["negative_slopes_by_teacher"] == {"2": 12, "3": 12}
    assert summary["anchor_emulated_roots_B72_by_teacher"] == {"2": 12, "3": 12}
    assert summary["cell_state_counts"] == {
        "FEASIBLE": 26,
        "HISTORY_MISSING": 91,
        "MAP_NOT_REPRESENTED": 1,
        "UNRESOLVED": 14,
    }
    assert len(cells) == 132
    assert len(scaling) == 24
    assert scaling.groupby("teacher_n_qubits")["slope"].max().lt(0.0).all()
    assert cells["median_history_nrmse"].min() == pytest.approx(
        0.0011021884945407998
    )
    assert cells["median_history_nrmse"].max() == pytest.approx(
        0.8741216887347973
    )


def test_committed_dense_analysis_matches_the_validated_aggregation() -> None:
    validated = validate_results(RESULTS, CONFIG_PATH)
    cells, scaling, summary = aggregate(validated, CONFIG_PATH)
    committed_cells = pd.read_csv(ANALYSIS / "dense_phase_cells.csv")
    committed_scaling = pd.read_csv(ANALYSIS / "dense_phase_root_scaling.csv")
    committed_summary = json.loads(
        (ANALYSIS / "dense_phase_summary.json").read_text(encoding="utf-8")
    )
    pd.testing.assert_frame_equal(
        committed_cells,
        cells,
        check_dtype=False,
        rtol=1e-10,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        committed_scaling,
        scaling,
        check_dtype=False,
        rtol=1e-10,
        atol=1e-12,
    )
    assert committed_summary == summary
