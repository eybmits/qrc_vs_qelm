from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from render_pra_extension_figures import (
    student_architecture_figure,
    validate_architecture_figure_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = (
    REPO_ROOT
    / "pra_extension/outputs/student_architecture_full_exploratory_v1_12roots/"
    "student_architecture_results.csv"
)
ANALYSIS = (
    REPO_ROOT
    / "pra_extension/outputs/student_architecture_analysis_v1_strict/"
    "student_architecture_analysis.json"
)
SELECTED = (
    REPO_ROOT
    / "pra_extension/outputs/student_architecture_analysis_v1_strict/"
    "student_architecture_selected_effects.csv"
)


def _inputs():
    return (
        pd.read_csv(RAW),
        pd.read_csv(SELECTED),
        json.loads(ANALYSIS.read_text()),
    )


def test_architecture_figure_inputs_match_strict_analysis():
    validate_architecture_figure_inputs(*_inputs())


def test_architecture_figure_fails_closed_on_selection_mutation():
    raw, selected, analysis = _inputs()
    selected.loc[0, "selected_architecture"] = "dense"
    with pytest.raises(AssertionError):
        validate_architecture_figure_inputs(raw, selected, analysis)


def test_architecture_figure_renders_pdf_and_png(tmp_path: Path):
    raw, selected, analysis = _inputs()
    output = tmp_path / "fig11_student_architecture_robustness"
    student_architecture_figure(raw, selected, analysis, output)
    assert output.with_suffix(".pdf").stat().st_size > 10_000
    # Font rasterization varies slightly across machines; this only guards
    # against an empty or obviously truncated render.
    assert output.with_suffix(".png").stat().st_size > 40_000
