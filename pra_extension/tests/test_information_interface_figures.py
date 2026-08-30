from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "pra_extension" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from render_information_interface_figures import (  # noqa: E402
    DEFAULT_DARK_SUMMARY,
    DEFAULT_PHASE_SOURCE,
    render_all,
    simplified_phase_table,
)


EXPECTED_ARTIFACTS = {
    "fig01_information_interface.pdf",
    "fig01_information_interface.png",
    "fig01_information_interface_source.json",
    "fig02_information_frontier.pdf",
    "fig02_information_frontier.png",
    "fig02_information_frontier_source.csv",
    "fig03_resource_diagnosis.pdf",
    "fig03_resource_diagnosis.png",
    "fig03_resource_diagnosis_source.csv",
}


def test_information_interface_figures_are_complete_traceable_and_deterministic(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = render_all(output_dir=first_dir)
    second = render_all(output_dir=second_dir)

    assert set(first) == EXPECTED_ARTIFACTS
    assert first == second
    ledger = json.loads(
        (first_dir / "figure_sha256.json").read_text(encoding="utf-8")
    )
    assert ledger == first
    assert "figure_sha256.json" not in ledger
    for name, expected in first.items():
        path = first_dir / name
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if name.endswith(".pdf"):
            assert path.stat().st_size > 10_000
        if name.endswith(".png"):
            assert path.stat().st_size > 50_000

    witness = json.loads(
        (first_dir / "fig01_information_interface_source.json").read_text(
            encoding="utf-8"
        )
    )
    assert witness == {
        "balanced_mse_floor": pytest.approx(0.003301905781096084),
        "fixed_pair_construction": (
            "two time-homogeneous input trajectories differ once in the addressable "
            "flag and share all later symmetry-preserving inputs"
            ),
            "output_gap": pytest.approx(0.11492442353296507),
            "positive_approximation_statement": (
                "continuous fading observable memory is uniformly approximable by a "
                "growing reset Bernstein QELM; contraction supplies explicit resources"
            ),
            "suffix_scope": "all allowed symmetry-preserving suffixes",
        "suffix_steps_numerically_checked": 80,
        "worst_case_absolute_error_floor": pytest.approx(0.057462211766482536),
    }

    phase = pd.read_csv(first_dir / "fig02_information_frontier_source.csv")
    assert len(phase) == 48
    assert phase["state"].value_counts().to_dict() == {
        "HISTORY_MISSING": 36,
        "FEASIBLE": 10,
        "UNRESOLVED": 2,
    }
    assert int(phase["any_root_disagreement"].sum()) == 11

    resources = pd.read_csv(first_dir / "fig03_resource_diagnosis_source.csv")
    assert len(resources) == 600
    assert resources["panel"].value_counts().to_dict() == {
        "history": 336,
        "shots": 192,
        "features": 72,
    }


def test_simplified_phase_table_preserves_registered_cell_grid() -> None:
    phase = simplified_phase_table(DEFAULT_PHASE_SOURCE)
    assert set(phase["teacher_n_qubits"]) == {2, 3}
    assert set(phase["teacher_gamma"]) == {0.0, 0.2, 0.5, 1.0, 2.0, 4.0}
    assert set(phase["window"]) == {1, 2, 4, 6}
    feasible = phase[phase["state"].eq("FEASIBLE")]
    counts = feasible.groupby(
        ["teacher_n_qubits", "minimum_feature_budget"]
    ).size()
    assert counts.to_dict() == {
        (2, 36.0): 3,
        (2, 72.0): 2,
        (3, 18.0): 1,
        (3, 36.0): 2,
        (3, 72.0): 2,
    }


def test_renderer_rejects_a_tampered_protected_memory_gap(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_DARK_SUMMARY.read_text(encoding="utf-8"))
    payload["expected_prefix_gap"] = 0.2
    tampered = tmp_path / "dark_sector_summary.json"
    tampered.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unexpected protected-memory output gap"):
        render_all(dark_summary_path=tampered, output_dir=tmp_path / "figures")
