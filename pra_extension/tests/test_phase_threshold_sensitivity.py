from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_phase_threshold_sensitivity import (  # noqa: E402
    DEFAULT_OUTPUT,
    EXPECTED_GAMMAS,
    _load_frozen_source,
    build_analysis,
    serialize_analysis,
    write_analysis,
)


PRIMARY_FRONTIER = [7, 7, 7, 7, 7, 6, 5, 4, 3, 3, 2]


def _frontier_values(record: dict[str, object]) -> list[int]:
    points = record["frontier"]
    assert isinstance(points, list)
    return [int(point["minimum_window_censored_at_7"]) for point in points]


def test_all_27_measured_frontiers_are_non_increasing_and_primary_is_exact() -> None:
    analysis = build_analysis()

    assert analysis["counts"] == {
        "input_rows": 23_760,
        "history_root_cells": 2_376,
        "history_root_cells_per_teacher_size": 792,
        "loss_window_cells": 198,
        "history_sufficiency_count_records": 594,
        "frontiers": 27,
        "frontier_loss_points": 297,
    }
    assert len(analysis["frontiers"]) == 27
    assert all(
        record["non_increasing_with_loss"] for record in analysis["frontiers"]
    )

    sensitivity = analysis["sensitivity_summary"]
    assert sensitivity["frontiers_evaluated"] == 27
    assert sensitivity["non_increasing_frontiers"] == 27
    assert sensitivity["all_27_measured_frontiers_non_increasing_with_loss"] is True
    assert sensitivity["alternative_frontier_points_compared_with_primary"] == 264
    assert sensitivity["alternative_frontier_point_shifts_relative_to_primary"] == {
        "shorter": 12,
        "same": 227,
        "longer": 25,
    }
    assert sensitivity[
        "teacher_size_loss_points_stable_across_all_nine_threshold_rules"
    ] == 17
    assert sensitivity["teacher_size_loss_points_changed_across_threshold_grid"] == 16

    primary = analysis["primary_frontiers"]
    assert [record["teacher_n_qubits"] for record in primary] == [2, 3, 4]
    assert all(record["tolerance_nrmse"] == 0.10 for record in primary)
    assert all(record["minimum_successful_roots"] == 10 for record in primary)
    assert all(_frontier_values(record) == PRIMARY_FRONTIER for record in primary)
    assert all(
        [point["teacher_gamma"] for point in record["frontier"]]
        == list(EXPECTED_GAMMAS)
        for record in primary
    )

    assert "does not establish invariance of individual cell labels" in analysis[
        "claim_boundary"
    ]
    assert "no tested window d<=6" in analysis["claim_boundary"]


def test_committed_threshold_sensitivity_json_is_a_deterministic_fresh_build(
    tmp_path: Path,
) -> None:
    fresh = build_analysis()
    committed = DEFAULT_OUTPUT.read_text(encoding="utf-8")
    assert committed == serialize_analysis(fresh)
    assert json.loads(committed) == fresh

    regenerated = tmp_path / "phase_threshold_sensitivity.json"
    written = write_analysis(regenerated)
    assert written == fresh
    assert regenerated.read_bytes() == DEFAULT_OUTPUT.read_bytes()


def test_source_validation_fails_closed_when_a_required_column_is_missing(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "suffix_decomposition.csv"
    malformed.write_text(
        "root_id,teacher_n_qubits,teacher_dissipator,teacher_gamma,window,model\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing columns.*history_nrmse"):
        _load_frozen_source(malformed, source_name="n4")
