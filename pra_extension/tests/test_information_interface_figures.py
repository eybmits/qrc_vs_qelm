from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "pra_extension" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from figure1_timeseries import output_provenance  # noqa: E402
from render_active_information_interface import (  # noqa: E402
    active_figure1_provenance as input_provenance,
    render_figure_1 as render_active_figure_1,
)
from render_information_interface_figures import (  # noqa: E402
    DEFAULT_DARK_SUMMARY,
    DEFAULT_DENSE_PHASE_SOURCE,
    DEFAULT_DENSE_SUFFIX_SOURCE,
    DEFAULT_DENSE_N4_PHASE_SOURCE,
    DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    DEFAULT_PHASE_SOURCE,
    DEFAULT_SCALING_SOURCE,
    DEFAULT_SHOT_SOURCE,
    DEFAULT_SUFFIX_SOURCE,
    COMPARISON_LAYOUT,
    FIGURE_SIZES,
    PALETTE,
    STYLE,
    _base_style,
    _resource_diagnosis_axes,
    _validated_dense_phase_source,
    _validated_dense_n4_phase_source,
    _validated_dense_merged_source,
    render_all,
    render_figure_2,
    render_figure_3,
    simplified_phase_table,
)


EXPECTED_ARTIFACTS = {
    "fig01_information_interface.pdf",
    "fig01_information_interface.png",
    "fig01_information_interface_render_source.json",
    "fig02_information_frontier.pdf",
    "fig02_information_frontier.png",
    "fig02_information_frontier_source.csv",
    "fig03_resource_diagnosis.pdf",
    "fig03_resource_diagnosis.png",
    "fig03_resource_diagnosis_source.csv",
}


EXPECTED_STYLE = {
    "font_family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "font_size": 9.5,
    "axes_labelsize": 10.0,
    "axes_titlesize": 10.5,
    "tick_labelsize": 9.0,
    "legend_fontsize": 8.5,
    "axes_linewidth": 0.55,
    "primary_linewidth": 1.1,
    "grid_linewidth": 0.35,
}

EXPECTED_FIGURE_SIZES = {
    "fig1": (7.055, 9.6 / 2.54),
    "fig2": (7.055, 2.65),
    "fig3": (7.055, 2.40),
}

EXPECTED_PALETTE = {
    "ink": "#20252B",
    "annotation": "#8B9198",
    "grid": "#DEE0E3",
    "neutral": "#F1F2F3",
    "history_missing": "#8B9198",
    "frozen": "#3868D9",
    "frozen_light": "#E7EDFA",
    "onset": "#D66A56",
    "onset_light": "#FAECE9",
    "scrambled": "#7C5A9F",
    "scrambled_light": "#EAE5F0",
    "yellow": "#FFC943",
    "pale_yellow": "#FFDD92",
    "orange": "#D66A56",
    "purple": "#7C5A9F",
    "viridis_18": "#3868D9",
    "viridis_36": "#D66A56",
    "viridis_72": "#7C5A9F",
    "family_2": "#3868D9",
    "family_3": "#7C5A9F",
    "story_ink": "#20252B",
    "story_border": "#DEE0E3",
    "story_neutral": "#F1F2F3",
    "story_blue": "#3868D9",
    "story_blue_light": "#E7EDFA",
    "story_green": "#D66A56",
    "story_green_light": "#FAECE9",
    "story_orange": "#7C5A9F",
    "story_orange_light": "#EAE5F0",
    "white": "#FFFFFF",
}


def test_resource_whiskers_preserve_root_variability_without_faint_traces(tmp_path, monkeypatch):
    from matplotlib.collections import PolyCollection
    from matplotlib.container import ErrorbarContainer
    from matplotlib.figure import Figure

    source = pd.read_csv(
        REPO_ROOT / "paper/figures/information_interface/fig03_resource_diagnosis_source.csv"
    )
    reviewed = []

    def inspect(fig, *args, **kwargs):
        if reviewed:
            return
        for index, (panel, xkey) in enumerate(
            (("history", "scaling_x"), ("features", "feature_budget"), ("shots", "total_shots"))
        ):
            ax = fig.axes[index]
            assert not any(isinstance(artist, PolyCollection) for artist in ax.collections)
            assert all(line.get_alpha() in (None, 1) for line in ax.lines)
            bars = [item for item in ax.containers if isinstance(item, ErrorbarContainer)]
            assert len(bars) == (1 if panel == "shots" else 3)
            for j, bar in enumerate(bars):
                rows = source[source["panel"].eq(panel)]
                if panel == "shots":
                    rows = rows[rows["metric"].eq("selected_policy_test_nrmse")]
                else:
                    rows = rows[rows["teacher_n_qubits"].eq(j + 2)]
                    if panel == "features":
                        rows = rows[rows["feature_budget"].le(72)]
                grouped = rows.groupby(xkey)["value"]
                x = np.array(sorted(rows[xkey].unique()))
                lo = grouped.quantile(0.10).to_numpy()
                hi = grouped.quantile(0.90).to_numpy()
                segments = bar.lines[2][0].get_segments()
                np.testing.assert_allclose(segments, np.array([
                    [[xx, lower], [xx, upper]] for xx, lower, upper in zip(x, lo, hi)
                ]))
                if panel != "shots":
                    np.testing.assert_allclose(
                        np.asarray(bar.lines[0].get_ydata(), dtype=float), grouped.median()
                    )
        reviewed.append(True)

    monkeypatch.setattr(Figure, "savefig", inspect)
    render_figure_3(DEFAULT_SCALING_SOURCE, DEFAULT_SUFFIX_SOURCE, DEFAULT_SHOT_SOURCE, tmp_path)
    assert reviewed == [True]


def test_resource_legends_clear_data_and_feature_crop_preserves_export(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from matplotlib.path import Path as MplPath

    reviewed = []

    def inspect(fig, *args, **kwargs):
        if reviewed:
            return
        fig.canvas.draw()
        for ax in fig.axes:
            # Include a small safety margin for marker/cap stroke widths.
            legend_box = ax.get_legend().get_window_extent().padded(2.0)
            for line in ax.lines:
                if line.get_marker() in {"o", "D", "s", "*", "_"}:
                    path = line.get_transform().transform_path(line.get_path())
                    assert not path.intersects_bbox(legend_box, filled=False)
                    assert not any(legend_box.contains(*point) for point in path.vertices)
            for collection in ax.collections:
                for segment in collection.get_segments():
                    path = collection.get_transform().transform_path(MplPath(segment))
                    assert not path.intersects_bbox(legend_box, filled=False)
        history_ax = fig.axes[0]
        assert history_ax.get_xlabel() == "window duration × loss rate"
        label_box = history_ax.xaxis.label.get_window_extent()
        assert fig.bbox.contains(label_box.x0, label_box.y0)
        assert fig.bbox.contains(label_box.x1, label_box.y1)
        assert not label_box.overlaps(fig.axes[1].yaxis.label.get_window_extent())
        history_legend_box = history_ax.get_legend().get_window_extent()
        target_y = history_ax.transData.transform((0, 0.10))[1]
        assert history_legend_box.y0 - target_y >= 0.03 * history_ax.bbox.height
        assert history_legend_box.y1 < history_ax.bbox.y1
        feature_ax = fig.axes[1]
        assert feature_ax.get_xscale() == "linear"
        np.testing.assert_array_equal(feature_ax.get_xticks(), [18, 24, 30, 36, 54, 72])
        boxes = [label.get_window_extent() for label in feature_ax.get_xticklabels()]
        assert all(a.x1 + 1 < b.x0 for a, b in zip(boxes, boxes[1:]))
        for line in feature_ax.lines:
            if line.get_marker() in {"o", "D", "s"}:
                np.testing.assert_array_equal(line.get_xdata(), [18, 24, 30, 36, 54, 72])
        shot_ax = fig.axes[2]
        target = next(t for t in shot_ax.texts if t.get_text() == "10% target")
        assert target.get_position() == (300, 0.097)
        assert target.get_fontsize() == 9.0 and target.get_va() == "top"
        label_box = target.get_window_extent().padded(0.5)
        assert shot_ax.bbox.contains(label_box.x0, label_box.y0)
        assert label_box.y1 < shot_ax.transData.transform((300, 0.10))[1]
        for line in shot_ax.lines:
            if line.get_marker() in {"o", "D", "s", "*", "_"}:
                path = line.get_transform().transform_path(line.get_path())
                assert not path.intersects_bbox(label_box, filled=False)
        for collection in shot_ax.collections:
            for segment in collection.get_segments():
                path = collection.get_transform().transform_path(MplPath(segment))
                assert not path.intersects_bbox(label_box, filled=False)
        reviewed.append(True)

    monkeypatch.setattr(Figure, "savefig", inspect)
    render_figure_3(DEFAULT_SCALING_SOURCE, DEFAULT_SUFFIX_SOURCE, DEFAULT_SHOT_SOURCE, tmp_path)
    assert reviewed == [True]
    exported = pd.read_csv(tmp_path / "fig03_resource_diagnosis_source.csv")
    assert len(exported) == 984
    assert set(exported.loc[exported["panel"].eq("features"), "feature_budget"]) == {
        18, 24, 30, 36, 54, 72, 108, 144,
    }


def test_all_resource_panels_have_matching_major_grids_below_data(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    reviewed = []

    def inspect(fig, *args, **kwargs):
        if reviewed:
            return
        fig.canvas.draw()
        assert len(fig.axes) == 3
        for ax in fig.axes:
            assert ax.get_axisbelow() is True
            for axis in (ax.xaxis, ax.yaxis):
                assert axis.get_zorder() < min(line.get_zorder() for line in ax.lines)
                guides = axis.get_gridlines()
                assert len(guides) >= 3
                assert all(guide.get_visible() for guide in guides)
                for guide in guides:
                    assert to_rgba(guide.get_color()) == to_rgba("#DEE0E3")
                    assert guide.get_alpha() == 1
                    assert guide.get_linewidth() == pytest.approx(.35)
                    assert guide.is_dashed()
                    assert guide._dash_pattern[0] == 0
                    assert guide._dash_pattern[1] == pytest.approx((2, 2))
                assert not any(tick.gridline.get_visible() for tick in axis.get_minor_ticks())
        reviewed.append(True)

    monkeypatch.setattr(Figure, "savefig", inspect)
    render_figure_3(DEFAULT_SCALING_SOURCE, DEFAULT_SUFFIX_SOURCE, DEFAULT_SHOT_SOURCE, tmp_path)
    assert reviewed == [True]


def test_resource_panels_share_one_row():
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=FIGURE_SIZES["fig3"])
    try:
        boxes = [ax.get_position() for ax in _resource_diagnosis_axes(fig)]
        assert len(boxes) == 3
        assert [b.y0 for b in boxes] == pytest.approx([boxes[0].y0] * 3)
        assert [b.height for b in boxes] == pytest.approx([boxes[0].height] * 3)
        assert [b.width for b in boxes] == pytest.approx([boxes[0].width] * 3)
        assert [b.width * fig.get_figwidth() for b in boxes] == pytest.approx([1.60] * 3)
        gaps = [(right.x0 - left.x1) * fig.get_figwidth() for left, right in zip(boxes, boxes[1:])]
        assert gaps == pytest.approx([COMPARISON_LAYOUT["panel_gap"]] * 2)
        assert all(gap > 0.5644 * 1.20 for gap in gaps)
        assert all(left.x1 < right.x0 for left, right in zip(boxes, boxes[1:]))
        assert all(0 <= b.x0 < b.x1 <= 1 and 0 <= b.y0 < b.y1 <= 1 for b in boxes)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("figure_number", (2, 3))
def test_quantitative_panels_are_physically_square_and_title_free(
    figure_number: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.figure import Figure

    captured = []

    def inspect_figure(fig: Figure, *args, **kwargs) -> None:
        if captured:
            return
        fig.canvas.draw()
        axes = fig.axes
        assert len(axes) == 3
        for index, ax in enumerate(axes):
            box = ax.get_window_extent()
            assert box.width / box.height == pytest.approx(1.0, rel=0.01)
            assert ax.get_box_aspect() == 1
            assert ax.get_title() == ""
            assert ax.get_title(loc="left") == ""
            assert ax.get_title(loc="right") == ""
            allowed_labels = {"a", "b", "c", "10% target"}
            if figure_number == 2:
                allowed_labels |= {f"$d={window}$" for window in range(1, 7)}
                allowed_labels.add(f"{index + 2}-qubit QRC")
            assert {text.get_text() for text in ax.texts} <= allowed_labels
            letter = [text for text in ax.texts if text.get_text() == "abc"[index]]
            assert len(letter) == 1
            assert letter[0].get_transform() == ax.transAxes
            assert all(0 < coordinate < 1 for coordinate in letter[0].get_position())
            assert all(spine.get_visible() for spine in ax.spines.values())
            legend = ax.get_legend()
            assert legend is not None and not legend.get_title().get_text()
            legend_box = legend.get_window_extent()
            assert box.contains(legend_box.x0, legend_box.y0)
            assert box.contains(legend_box.x1, legend_box.y1)
        assert not fig.legends
        assert not fig.texts
        # Both figures occupy one row, with unscaled square plotting boxes.
        boxes = [ax.get_window_extent() for ax in axes]
        assert [box.y0 for box in boxes] == pytest.approx([boxes[0].y0] * 3)
        assert all(left.x1 < right.x0 for left, right in zip(boxes, boxes[1:]))
        if figure_number == 3:
            # Check white gutters after the next panel's full y-axis label,
            # not only the distance between the rectangular frames.
            for left, right in zip(axes, axes[1:]):
                gap = right.yaxis.label.get_window_extent().x0 - left.get_window_extent().x1
                assert gap * 72 / fig.dpi >= 6.0
            for ax in axes:
                assert ax.xaxis.label.get_fontsize() == STYLE["axes_labelsize"]
                assert ax.yaxis.label.get_fontsize() == STYLE["axes_labelsize"]
                assert ax.xaxis.labelpad == 4.0 and ax.yaxis.labelpad == 3.0
        captured.append(fig)

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    if figure_number == 2:
        render_figure_2(
            DEFAULT_DENSE_PHASE_SOURCE,
            DEFAULT_DENSE_SUFFIX_SOURCE,
            DEFAULT_DENSE_N4_PHASE_SOURCE,
            DEFAULT_DENSE_N4_SUFFIX_SOURCE,
            tmp_path,
        )
    else:
        render_figure_3(
            DEFAULT_SCALING_SOURCE,
            DEFAULT_SUFFIX_SOURCE,
            DEFAULT_SHOT_SOURCE,
            tmp_path,
        )
    assert len(captured) == 1


def test_integrated_frontier_markers_preserve_every_measurement_and_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.figure import Figure

    phase = _validated_dense_merged_source(
        DEFAULT_DENSE_PHASE_SOURCE,
        DEFAULT_DENSE_SUFFIX_SOURCE,
        DEFAULT_DENSE_N4_PHASE_SOURCE,
        DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    )
    outcome_styles = {
        "HISTORY_NOT_ROBUST": ("o", False),
        "NO_TESTED_QELM_ROBUST": ("x", True),
        "QELM_18": ("o", True),
        "QELM_36": ("s", True),
        "QELM_72": ("D", True),
    }
    drawn = {}

    def inspect_figure(fig: Figure, *args, **kwargs) -> None:
        if drawn:
            return
        assert len(fig.axes) == 3
        for n_qubits, ax in zip((2, 3, 4), fig.axes):
            family = phase.loc[phase["teacher_n_qubits"].eq(n_qubits)]
            curves = [line for line in ax.lines if len(line.get_xdata()) == 11
                      and line.get_marker() in (None, "None", "")]
            assert len(curves) == 6
            for window, curve in zip(range(1, 7), curves):
                expected = family.loc[family["window"].eq(window)].sort_values("teacher_gamma")
                assert list(curve.get_xdata()) == pytest.approx(expected["teacher_gamma"])
                assert list(curve.get_ydata()) == pytest.approx(expected["median_history_nrmse"])
                color = to_rgba(curve.get_color())
                markers = [line for line in ax.lines
                           if line.get_marker() in {"o", "x", "s", "D"}
                           and to_rgba(line.get_markeredgecolor()) == color]
                assert sum(len(line.get_xdata()) for line in markers) == 11
                for marker in markers:
                    for gamma, error in zip(marker.get_xdata(), marker.get_ydata()):
                        row = expected.loc[expected["teacher_gamma"].eq(gamma)].iloc[0]
                        assert float(error) == pytest.approx(row["median_history_nrmse"])
                        outcome = str(row["display_outcome"])
                        shape, filled = outcome_styles[outcome]
                        assert marker.get_marker() == shape
                        assert to_rgba(marker.get_markerfacecolor()) == (
                            color if filled else to_rgba("white")
                        )
                        key = (n_qubits, float(gamma), window)
                        assert key not in drawn
                        drawn[key] = outcome

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    render_figure_2(
        DEFAULT_DENSE_PHASE_SOURCE, DEFAULT_DENSE_SUFFIX_SOURCE,
        DEFAULT_DENSE_N4_PHASE_SOURCE, DEFAULT_DENSE_N4_SUFFIX_SOURCE, tmp_path,
    )
    assert len(drawn) == 198
    expected_outcomes = phase.set_index(
        ["teacher_n_qubits", "teacher_gamma", "window"],
    )["display_outcome"].to_dict()
    assert drawn == expected_outcomes
    # A sub-target median is not sufficient for the frozen 10-of-12 rule.
    below_but_not_robust = phase.loc[
        phase["median_history_nrmse"].le(0.10) & phase["history_sufficient_n"].lt(10)
    ]
    assert len(below_but_not_robust) == 6
    for row in below_but_not_robust.itertuples():
        assert drawn[(row.teacher_n_qubits, row.teacher_gamma, row.window)] == "HISTORY_NOT_ROBUST"
    # The displayed symbols must not turn real d=5 success / d=6 failure into
    # a monotonic feasible region, even though history error decreases.
    for n_qubits, successful in ((2, "QELM_72"), (3, "QELM_72"), (4, "QELM_36")):
        assert drawn[(n_qubits, 2.0, 5)] == successful
        assert drawn[(n_qubits, 2.0, 6)] == "NO_TESTED_QELM_ROBUST"
    archived = REPO_ROOT / "paper/figures/information_interface/fig02_information_frontier_source.csv"
    assert (tmp_path / archived.name).read_bytes() == archived.read_bytes()


def _pdf_media_box(path: Path) -> tuple[float, float]:
    """Return the first PDF page width and height in PostScript points."""

    # TikZ stores the page object in a compressed stream; inspect the PDF using
    # Poppler rather than relying on Matplotlib's uncompressed object layout.
    info = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    ).stdout
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    match = re.search(
        r"Page size:\s*(" + number + r")\s*x\s*(" + number + r")\s+pts",
        info,
    )
    if match is None:
        raise AssertionError(f"{path.name} has no readable page dimensions")
    width, height = (float(value) for value in match.groups())
    return width, height


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
    assert matplotlib.rcParams["axes.labelweight"] == "bold"


def test_native_tikz_story_preserves_readable_labels_and_emulation_scope() -> None:
    tikz_source = (
        REPO_ROOT / "paper" / "figures" / "information_interface"
        / "fig01_information_interface.tex"
    ).read_text(encoding="utf-8")
    assert r"\usepackage{helvet}" in tikz_source
    assert r"\renewcommand{\familydefault}{\sfdefault}" in tikz_source
    core, external_labels = tikz_source.split("% Case labels sit outside the data frames", 1)
    font_sizes = [float(size) for size in re.findall(r"\\fontsize\{([\d.]+)\}", core)]
    assert font_sizes and min(font_sizes) >= 10 and max(font_sizes) <= 12
    assert [float(size) for size in re.findall(r"\\fontsize\{([\d.]+)\}", external_labels)] == [9, 9]
    assert r"\resizebox" not in tikz_source and r"\scalebox" not in tikz_source
    for required_component in (
        "QRC", "QELM", r"\rho", r"\widehat", "reset", "window",
        "carried state", "E(S_d)", "fixed gates", "observables",
    ):
        assert required_component in tikz_source
    assert tikz_source.count("{observables}") == 2
    assert r"{Fading\\memory}" not in tikz_source
    assert r"{Persistent\\memory}" not in tikz_source
    assert "measured features" not in tikz_source
    artwork = re.sub(r"%[^\n]*", "", tikz_source)
    assert not re.search(
        r"\\node[^;]*\{[^{}]*(?:trained|readout)[^{}]*\};", artwork, re.I,
    )
    triangles = re.findall(r"\\draw\[[^\]]*fill=gold[^\]]*\][^;]*--cycle;", artwork)
    assert len(triangles) == 2
    assert r"{$y_t$}" in artwork and r"{$\widehat y_t$}" in artwork
    assert "declared output" not in artwork and "emulated output" not in artwork
    assert r"\mathcal C" not in artwork
    assert "schematic" in tikz_source.lower()
    for obsolete in ("best history error", "desired error tolerance", r"D_d/2", r"\varepsilon"):
        assert obsolete not in artwork
    # The two traces illustrate access to one old bit at fixed context,
    # while the render record keeps the separate uniform-theorem scope.
    output = output_provenance()
    assert "not simulated, fitted, or measured QRC/QELM performance" in output["status"]
    assert "not labeled or identified as the generic supremal D_d/2" in output["curve_gap_scope"]
    assert "not a fitted finite-feature QELM" in output["emulator_scope"]
    assert "not the uniform D_d limit" in output["cases"]["fading"]["decay_scope"]

def test_plot_labels_use_plain_language_and_separate_resource_questions() -> None:
    renderer_source = (
        CODE_ROOT / "render_information_interface_figures.py"
    ).read_text(encoding="utf-8")
    assert r"\mathbf" not in renderer_source
    assert "scaled context" not in renderer_source
    assert 'ax.set_xlabel("window duration × loss rate")' in renderer_source
    evaluation = " ".join(
        (REPO_ROOT / "paper/icaart2027/sections/evaluation.tex").read_text().split()
    )
    # The active atlas separates window length from loss instead of
    # requiring readers to interpret the archived product-coordinate plot.
    atlas_source = (
        REPO_ROOT / "paper/figure2_candidates/render_resource_atlas.py"
    ).read_text(encoding="utf-8")
    assert 'ax.set_xlabel("Inputs per window"' in atlas_source
    # The readability rewrite moves the full design into the methods appendix,
    # while retaining the tested-range qualifications in the main text.
    methods = " ".join(
        (REPO_ROOT / "paper/icaart2027/sections/methodology.tex").read_text().split()
    )
    assert "target reservoirs have two, three, or four qubits" in methods
    assert r"\{0,0.2,0.5,0.75,1,1.25,1.5,2,2.5,3,4\}" in methods
    assert "windows range from one to six" in methods
    assert "two inputs meet the history criterion across all three sizes" in evaluation
    assert "At weak loss, no tested window of up to six inputs does" in evaluation
    assert "longer windows remain unresolved" in evaluation
    assert '"?"' not in renderer_source
    assert 'label="root split"' not in renderer_source
    assert 'label="unresolved"' not in renderer_source
    assert 'label="map limited"' not in renderer_source
    for removed_row_title in (
        "How much old history remains?",
        "Which reset-QELM reaches the target?",
        "Smallest successful feature count",
        "Include enough",
        "Add enough",
        "Resolve features",
        'f"{n_qubits}-qubit QRC", y=',
    ):
        assert removed_row_title not in renderer_source
    for required_plot_label in (
        "QELM unmet",
        "history unmet",
    ):
        assert required_plot_label in renderer_source


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
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr("render_information_interface_figures.render_figure_1", render_active_figure_1)
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

    story = json.loads(
        (first_dir / "fig01_information_interface_render_source.json").read_text(
            encoding="utf-8"
        )
    )
    assert story["figure_question"] == (
        "when can explicit finite context reproduce recurrent input-output behavior"
    )
    shared_input = story["input_timeseries"]
    assert shared_input == input_provenance(
        REPO_ROOT / "paper/figures/information_interface/fig01_information_interface.tex"
    )
    assert shared_input["plot_geometry"]["input_axis_count"] == 1
    assert shared_input["plot_geometry"]["sample_axes_side_cm"] == 4.15
    assert shared_input["plot_geometry"]["full_input_axes_bounds_cm"] == [1.045, 6.725, 5.195, 10.875]
    assert "no separate duplicate plot" in shared_input["plot_geometry"]["window_view"]
    assert shared_input["plot_geometry"]["window_bracket_index_extent"] == [140.5, 180]
    assert shared_input["window_indices"] == list(range(141, 181))
    assert shared_input["future_context_indices"] == list(range(181, 201))
    assert set(shared_input["window_indices"]).isdisjoint(shared_input["future_context_indices"])
    assert story["selected_design"] == "variant_a_architectures"
    assert "selected_theory_panel" not in story
    assert "panel_b_history_error" not in story
    assert "history_error_curve" not in shared_input
    assert story["removed_history_error_panel"]["status"] == "removed from the artwork"
    assert story["panel_letters"] == []
    historical = story["historical_input_audit"]
    historical_path = REPO_ROOT / historical["path"]
    assert historical_path.name == "fig01_information_interface_source.json"
    assert historical["sha256"] == (
        "d5a76f449661e3f4fabf909a6271208f838907f076db1c02652b771665d23965"
    )
    assert hashlib.sha256(historical_path.read_bytes()).hexdigest() == historical["sha256"]
    assert "not the source description of the current drawing" in historical["status"]
    # Rendering a current drawing must never rewrite the N5-bound snapshot.
    assert not (first_dir / historical_path.name).exists()
    assert not (second_dir / historical_path.name).exists()
    assert "fixed target readout" in story["readout_scope"]
    assert "trained reset-QELM emulator readout" in story["readout_scope"]
    assert "matching yellow observable banks" in story["architecture_motifs"]
    assert story["canvas_aspect_ratio"] == pytest.approx(17.9197 / 9.6)
    assert story["canvas_bounds_cm"] == [0, 3.6, 17.9197, 13.2]
    assert set(story["panels"]) == {"architecture", "fading", "persistent"}
    assert "recurrent" in story["panels"]["architecture"]
    assert "reset" in story["panels"]["architecture"]
    assert "upper output frame" in story["panels"]["fading"]
    assert "lower output frame" in story["panels"]["persistent"]
    theory = story["theorem_context"]
    assert "future declared outputs" in theory["memory_relevance"]
    assert "full-state forgetting is not required" in theory["memory_relevance"]
    assert "supremal declared-output distance" in theory["common_suffix_diameter"]
    assert "D_d/2" in theory["scalar_information_limit"]
    assert "worst-case absolute scalar error" in theory["scalar_information_limit"]
    assert "not a plotted curve or measured property" in theory["status"]
    assert "not generic D_d/2 curves" in theory["illustration_boundary"]
    assert "fixed d does not establish uniform convergence" in theory["illustration_boundary"]
    assert "panel_c_output_traces" not in story
    assert "panel_a_output_comparisons" not in story
    output_traces = story["output_comparisons"]
    assert output_traces == shared_input["output_comparisons"]
    assert output_traces["axis_count"] == 2
    assert output_traces["axes_size_cm"] == [4.15, 3.2]
    assert output_traces["axes_bounds_cm"] == {
        "fading": [13.525, 8.58, 17.675, 11.78],
        "persistent": [13.525, 4.99, 17.675, 8.19],
    }
    assert output_traces["amplitude_limits"] == [0.45, 1.35]
    assert output_traces["amplitude_ticks"] == [0.6, 0.8, 1.0, 1.2]
    assert output_traces["time_limits"] == [-3, 103]
    assert output_traces["time_ticks"] == [0, 40, 80]
    assert output_traces["time_tick_labels"] == ["0", "d", "2d"]
    assert output_traces["template_stream"]["seed"] == 2
    assert output_traces["template_stream"]["generated_count"] == 101
    assert "not simulated, fitted, or measured QRC/QELM performance" in output_traces["status"]
    assert "separate auxiliary toy timeline" in output_traces["relationship_to_input"]
    assert "same fixed d=40" in output_traces["timeline_scope"]
    assert output_traces["memory_access"]["first_excluded_step"] == 40
    assert output_traces["case_order"] == ["fading", "persistent"]
    cases = output_traces["cases"]
    assert not cases["fading"]["time_tick_labels_visible"]
    assert cases["persistent"]["time_tick_labels_visible"]
    persistent_error = np.asarray(cases["persistent"]["post_expiry_pair_midpoint_error_values"])
    assert persistent_error[0] == pytest.approx(0.0107421875)
    assert np.all(np.diff(persistent_error) >= 0)
    np.testing.assert_array_equal(persistent_error[56:], np.full(5, 0.25))
    fading_error = np.asarray(cases["fading"]["post_expiry_pair_midpoint_error_values"])
    np.testing.assert_array_equal(fading_error, np.zeros(61))
    assert cases["fading"]["forgetting_horizon_steps"] == 40
    assert "special illustrative case, not a generic property of fading memory" in cases["fading"]["decay_scope"]
    assert "not a fitted finite-feature QELM" in output_traces["emulator_scope"]
    assert "not an independent third forecasting target" in output_traces["target_definition"]
    assert output_traces["legend_labels"] == ["QRC", "QELM"]
    assert output_traces["legend_geometry"]["bounds_cm"] == [14.00, 8.11, 17.27, 8.64]
    assert output_traces["shared_time_label_anchor_cm"] is None
    assert output_traces["column_order"] == ["target", "prediction"]
    tikz_path = (
        REPO_ROOT / "paper" / "figures" / "information_interface"
        / "fig01_information_interface.tex"
    )
    assert story["tikz_source_sha256"] == hashlib.sha256(tikz_path.read_bytes()).hexdigest()
    assert "native" in story["typography"]
    assert "7.055" in story["typography"]
    assert story["schematic_curves"] is True
    assert theory["regimes"]["finite"]["condition"] == "D_d = 0 for some finite d"
    assert "D_d tends to zero" in theory["regimes"]["fading"]["condition"]
    assert "r_vis < 1" in theory["regimes"]["fading"]["condition"]
    assert "without a guarantee of exact emulation" in theory["regimes"]["fading"]["consequence"]
    assert "Delta/2" in theory["regimes"]["persistent"]["consequence"]
    assert "D_d tends to zero" in theory["emulation_statement"]
    assert "exact-expectation" in theory["emulation_statement"]
    assert "universal on each finite" in theory["emulation_statement"]
    for required_scope in (
        "finite-dimensional", "input-continuous", "fixed finite",
        "compact inputs", "freely concatenable suffixes",
    ):
        assert required_scope in theory["scope"]
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        story["upstream_dark_sector_summary_sha256"],
    )
    assert not {
        "balanced_mse_floor",
        "fixed_pair_construction",
        "output_gap",
        "suffix_scope",
    }.intersection(story)

    phase = pd.read_csv(first_dir / "fig02_information_frontier_source.csv")
    assert len(phase) == 198
    assert phase["evidence_cohort"].value_counts().to_dict() == {
        "frozen_dense_n2_n3": 132,
        "prospective_dense_n4_followup": 66,
    }
    assert phase["state"].value_counts().to_dict() == {
        "HISTORY_MISSING": 138,
        "FEASIBLE": 38,
        "UNRESOLVED": 21,
        "MAP_NOT_REPRESENTED": 1,
    }
    assert int(phase["any_root_disagreement"].sum()) == 77
    assert (phase["history_n_roots"] == 12).all()
    assert int(phase["history_n_roots"].sum()) == 2_376
    assert phase["display_outcome"].value_counts().to_dict() == {
        "HISTORY_NOT_ROBUST": 141,
        "QELM_36": 28,
        "NO_TESTED_QELM_ROBUST": 19,
        "QELM_18": 7,
        "QELM_72": 3,
    }
    assert (phase["history_root_count"] == 12).all()
    assert int(phase["history_root_count"].sum()) == 2_376
    assert phase["median_history_nrmse"].min() == pytest.approx(
        0.00110218849454
    )
    assert phase["median_history_nrmse"].max() == pytest.approx(
        0.874121688735
    )

    resources = pd.read_csv(first_dir / "fig03_resource_diagnosis_source.csv")
    assert len(resources) == 984
    assert resources["panel"].value_counts().to_dict() == {
        "history": 504,
        "shots": 192,
        "features": 288,
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


def test_dense_phase_source_preserves_fresh_complete_grid() -> None:
    phase = _validated_dense_phase_source(
        DEFAULT_DENSE_PHASE_SOURCE,
        DEFAULT_DENSE_SUFFIX_SOURCE,
    )
    assert set(phase["teacher_n_qubits"]) == {2, 3}
    assert set(phase["teacher_gamma"]) == {
        0.0,
        0.2,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
    }
    assert set(phase["window"]) == {1, 2, 3, 4, 5, 6}
    assert phase["run_manifest_sha256"].nunique() == 1
    assert phase["root_set_sha256"].nunique() == 1
    assert phase.loc[
        phase["state"].eq("FEASIBLE"),
        ["emulated_n_B18", "emulated_n_B36", "emulated_n_B72"],
    ].max(axis=1).ge(10).all()
    robust_history = phase["history_sufficient_n"].ge(10)
    assert phase.loc[robust_history, "display_outcome"].ne(
        "HISTORY_NOT_ROBUST"
    ).all()
    assert phase.loc[~robust_history, "display_outcome"].eq(
        "HISTORY_NOT_ROBUST"
    ).all()
    excluded = phase[
        phase["teacher_n_qubits"].eq(3)
        & phase[["teacher_gamma", "window"]].apply(
            tuple,
            axis=1,
        ).isin({(1.0, 6), (1.25, 5), (1.5, 4)})
    ]
    assert excluded["display_outcome"].eq("HISTORY_NOT_ROBUST").all()


def test_dense_n4_source_preserves_separate_prospective_grid() -> None:
    phase = _validated_dense_n4_phase_source(
        DEFAULT_DENSE_N4_PHASE_SOURCE,
        DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    )
    assert len(phase) == 66
    assert set(phase["teacher_n_qubits"]) == {4}
    assert set(phase["teacher_gamma"]) == {
        0.0,
        0.2,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
    }
    assert set(phase["window"]) == {1, 2, 3, 4, 5, 6}
    assert phase["evidence_cohort"].eq("prospective_dense_n4_followup").all()
    assert phase["history_root_count"].eq(12).all()
    assert int(phase["history_root_count"].sum()) == 792
    assert phase["display_outcome"].value_counts().to_dict() == {
        "HISTORY_NOT_ROBUST": 47,
        "QELM_36": 10,
        "NO_TESTED_QELM_ROBUST": 7,
        "QELM_18": 2,
    }


def test_merged_dense_source_keeps_cohort_provenance_separate() -> None:
    phase = _validated_dense_merged_source(
        DEFAULT_DENSE_PHASE_SOURCE,
        DEFAULT_DENSE_SUFFIX_SOURCE,
        DEFAULT_DENSE_N4_PHASE_SOURCE,
        DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    )
    assert len(phase) == 198
    assert not phase.duplicated(
        ["teacher_n_qubits", "teacher_gamma", "window"]
    ).any()
    assert phase["evidence_cohort"].value_counts().to_dict() == {
        "frozen_dense_n2_n3": 132,
        "prospective_dense_n4_followup": 66,
    }
    for cohort in phase["evidence_cohort"].unique():
        selected = phase[phase["evidence_cohort"].eq(cohort)]
        assert selected["run_manifest_sha256"].nunique() == 1
        assert selected["root_set_sha256"].nunique() == 1
        assert selected["config_sha256"].nunique() == 1
    assert phase.groupby("evidence_cohort")["config_sha256"].first().nunique() == 2
    assert phase.groupby("evidence_cohort")["root_set_sha256"].first().nunique() == 2


def test_integrated_frontier_retains_uniformly_history_insufficient_low_loss_cells() -> None:
    phase = _validated_dense_merged_source(
        DEFAULT_DENSE_PHASE_SOURCE,
        DEFAULT_DENSE_SUFFIX_SOURCE,
        DEFAULT_DENSE_N4_PHASE_SOURCE,
        DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    )
    low_loss = phase.loc[phase["teacher_gamma"].le(1.0)]
    assert len(low_loss) == 90
    assert set(low_loss["teacher_n_qubits"]) == {2, 3, 4}
    assert set(low_loss["window"]) == {1, 2, 3, 4, 5, 6}
    assert low_loss["display_outcome"].eq("HISTORY_NOT_ROBUST").all()
    assert low_loss["history_sufficient_n"].lt(10).all()
    # Both measurements and status symbols now remain in the integrated curves.
    archived = pd.read_csv(
        REPO_ROOT / "paper" / "figures" / "information_interface"
        / "fig02_information_frontier_source.csv"
    )
    assert len(archived.loc[archived["teacher_gamma"].le(1.0)]) == len(low_loss)


def test_feature_frontier_keeps_real_nonmonotonic_window_outcomes() -> None:
    phase = _validated_dense_merged_source(
        DEFAULT_DENSE_PHASE_SOURCE,
        DEFAULT_DENSE_SUFFIX_SOURCE,
        DEFAULT_DENSE_N4_PHASE_SOURCE,
        DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    )
    cells = phase.loc[phase["teacher_gamma"].eq(2.0) & phase["window"].isin((5, 6))]
    assert len(cells) == 6
    assert cells["history_sufficient_n"].eq(12).all()
    outcomes = cells.set_index(["teacher_n_qubits", "window"])["display_outcome"]
    assert outcomes.to_dict() == {
        (2, 5): "QELM_72",
        (2, 6): "NO_TESTED_QELM_ROBUST",
        (3, 5): "QELM_72",
        (3, 6): "NO_TESTED_QELM_ROBUST",
        (4, 5): "QELM_36",
        (4, 6): "NO_TESTED_QELM_ROBUST",
    }
    archived = pd.read_csv(
        REPO_ROOT / "paper" / "figures" / "information_interface"
        / "fig02_information_frontier_source.csv"
    )
    archived_cells = archived.loc[
        archived["teacher_gamma"].eq(2.0) & archived["window"].isin((5, 6))
    ]
    assert archived_cells.set_index(
        ["teacher_n_qubits", "window"]
    )["display_outcome"].to_dict() == outcomes.to_dict()


def test_dense_phase_validator_rejects_missing_cell_and_mixed_run(tmp_path: Path) -> None:
    phase = pd.read_csv(DEFAULT_DENSE_PHASE_SOURCE)
    missing = tmp_path / "missing.csv"
    phase.iloc[:-1].to_csv(missing, index=False)
    with pytest.raises(ValueError, match="132 unique cells"):
        _validated_dense_phase_source(missing)

    mixed = phase.copy()
    mixed.loc[0, "run_manifest_sha256"] = "0" * 64
    mixed_path = tmp_path / "mixed.csv"
    mixed.to_csv(mixed_path, index=False)
    with pytest.raises(ValueError, match="run_manifest_sha256"):
        _validated_dense_phase_source(mixed_path)


def test_dense_n4_validator_rejects_missing_cell_and_mixed_run(tmp_path: Path) -> None:
    phase = pd.read_csv(DEFAULT_DENSE_N4_PHASE_SOURCE)
    missing = tmp_path / "missing_n4.csv"
    phase.iloc[:-1].to_csv(missing, index=False)
    with pytest.raises(ValueError, match="66 unique cells"):
        _validated_dense_n4_phase_source(missing)

    mixed = phase.copy()
    mixed.loc[0, "run_manifest_sha256"] = "0" * 64
    mixed_path = tmp_path / "mixed_n4.csv"
    mixed.to_csv(mixed_path, index=False)
    with pytest.raises(ValueError, match="run_manifest_sha256"):
        _validated_dense_n4_phase_source(mixed_path)


def test_dense_phase_validator_rejects_tampered_raw_suffix_source(
    tmp_path: Path,
) -> None:
    suffix = pd.read_csv(DEFAULT_DENSE_SUFFIX_SOURCE)
    suffix.loc[0, "history_nrmse"] = float(suffix.loc[0, "history_nrmse"]) + 0.01
    tampered = tmp_path / "suffix_decomposition.csv"
    suffix.to_csv(tampered, index=False)
    with pytest.raises(ValueError, match="registered SHA-256"):
        _validated_dense_phase_source(DEFAULT_DENSE_PHASE_SOURCE, tampered)


def test_dense_n4_validator_rejects_tampered_raw_suffix_source(
    tmp_path: Path,
) -> None:
    suffix = pd.read_csv(DEFAULT_DENSE_N4_SUFFIX_SOURCE)
    suffix.loc[0, "history_nrmse"] = float(suffix.loc[0, "history_nrmse"]) + 0.01
    tampered = tmp_path / "suffix_decomposition_n4.csv"
    suffix.to_csv(tampered, index=False)
    with pytest.raises(ValueError, match="registered SHA-256"):
        _validated_dense_n4_phase_source(DEFAULT_DENSE_N4_PHASE_SOURCE, tampered)


def test_renderer_rejects_a_tampered_protected_memory_gap(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_DARK_SUMMARY.read_text(encoding="utf-8"))
    payload["expected_prefix_gap"] = 0.2
    tampered = tmp_path / "dark_sector_summary.json"
    tampered.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unexpected protected-memory output gap"):
        render_all(dark_summary_path=tampered, output_dir=tmp_path / "figures")
