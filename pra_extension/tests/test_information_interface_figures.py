from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "pra_extension" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from render_information_interface_figures import (  # noqa: E402
    DEFAULT_DARK_SUMMARY,
    DEFAULT_PHASE_SOURCE,
    FIGURE_SIZES,
    PALETTE,
    STYLE,
    _base_style,
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


EXPECTED_STYLE = {
    "font_family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "font_size": 8.0,
    "axes_labelsize": 8.0,
    "axes_titlesize": 8.2,
    "tick_labelsize": 6.8,
    "legend_fontsize": 6.7,
    "axes_linewidth": 0.52,
    "primary_linewidth": 0.64,
    "grid_linewidth": 0.40,
}

EXPECTED_FIGURE_SIZES = {
    "fig1": (7.055, 4.05),
    "fig2": (7.055, 2.62),
    "fig3": (7.055, 2.55),
}

EXPECTED_PALETTE = {
    "ink": "#000000",
    "annotation": "#7F7F7F",
    "grid": "#B0B0B0",
    "neutral": "#F1F2F2",
    "history_missing": "#A7A9AC",
    "frozen": "#3B86FF",
    "frozen_light": "#EBF2FF",
    "onset": "#00A14B",
    "onset_light": "#E6F5EC",
    "scrambled": "#FC4D30",
    "scrambled_light": "#FEECE9",
    "yellow": "#FFC943",
    "pale_yellow": "#FFDD92",
    "orange": "#FF8431",
    "purple": "#A93BFF",
    "viridis_18": "#7AD151",
    "viridis_36": "#2A788E",
    "viridis_72": "#482475",
    "family_2": "#253494",
    "family_3": "#E8590C",
    "white": "#FFFFFF",
}


def _pdf_media_box(path: Path) -> tuple[float, float]:
    """Return the first PDF page width and height in PostScript points."""

    number = rb"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    match = re.search(
        rb"/MediaBox\s*\[\s*(" + number + rb")\s+(" + number + rb")\s+"
        rb"(" + number + rb")\s+(" + number + rb")\s*\]",
        path.read_bytes(),
    )
    if match is None:
        raise AssertionError(f"{path.name} has no directly encoded MediaBox")
    left, bottom, right, top = (float(value) for value in match.groups())
    return right - left, top - bottom


def test_reference_figure_style_contract_is_frozen() -> None:
    assert STYLE == EXPECTED_STYLE
    assert PALETTE == EXPECTED_PALETTE
    assert FIGURE_SIZES == EXPECTED_FIGURE_SIZES
    assert {width for width, _ in FIGURE_SIZES.values()} == {7.055}
    _base_style()
    assert list(matplotlib.rcParams["font.family"]) == STYLE["font_family"]
    assert matplotlib.rcParams["font.size"] == STYLE["font_size"]
    assert matplotlib.rcParams["axes.labelsize"] == STYLE["axes_labelsize"]
    assert matplotlib.rcParams["axes.titlesize"] == STYLE["axes_titlesize"]
    assert matplotlib.rcParams["xtick.labelsize"] == STYLE["tick_labelsize"]
    assert matplotlib.rcParams["ytick.labelsize"] == STYLE["tick_labelsize"]
    assert matplotlib.rcParams["legend.fontsize"] == STYLE["legend_fontsize"]
    assert matplotlib.rcParams["axes.linewidth"] == STYLE["axes_linewidth"]


def test_panel_labels_are_plain_helvetica_text_not_math_glyphs() -> None:
    renderer_source = (
        CODE_ROOT / "render_information_interface_figures.py"
    ).read_text(encoding="utf-8")
    assert r"\mathbf" not in renderer_source
    assert ".set_title(" not in renderer_source
    # Eight call sites plus the helper; the Fig. 2 call runs once for each panel.
    assert renderer_source.count("_panel_heading(") == 9
    assert set(
        re.findall(
            r"figsize\s*=\s*FIGURE_SIZES\[['\"](fig[123])['\"]\]",
            renderer_source,
        )
    ) == set(FIGURE_SIZES)


def test_committed_figure_pdfs_preserve_width_aspect_and_font_contract() -> None:
    figure_dir = REPO_ROOT / "paper" / "figures" / "information_interface"
    pdfs = {
        "fig1": figure_dir / "fig01_information_interface.pdf",
        "fig2": figure_dir / "fig02_information_frontier.pdf",
        "fig3": figure_dir / "fig03_resource_diagnosis.pdf",
    }
    for key, path in pdfs.items():
        payload = path.read_bytes()
        assert b"dejavu" not in payload.lower(), path.name
        width_points, height_points = _pdf_media_box(path)
        target_width, target_height = FIGURE_SIZES[key]
        assert width_points / 72 == pytest.approx(target_width, abs=0.005)
        assert height_points / 72 == pytest.approx(target_height, abs=0.005)
        assert width_points / height_points == pytest.approx(
            target_width / target_height,
            rel=0.005,
        )


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
    assert set(resources.loc[resources["panel"].eq("features"), "metric"]) == {
        "total_nrmse"
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
