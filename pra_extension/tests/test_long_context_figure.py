"""Presentation tests using synthetic fixtures, never fresh N5 results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "pra_extension" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from render_long_context_n5 import SERIES, _validate_feature_source, render  # noqa: E402
from render_information_interface_figures import (  # noqa: E402
    COMPARISON_LAYOUT, STYLE, _resource_diagnosis_axes,
)


WINDOWS = np.array([2, 4, 6, 8, 10], dtype=float)
FEATURE_BUDGETS = np.array([30, 60, 120], dtype=int)
COLORS = {"history": "#20252B", "qelm": "#D66A56", "quadratic": "#7C5A9F"}


@pytest.fixture
def synthetic_source() -> pd.DataFrame:
    """All required cells, with asymmetric artificial root variability."""
    root_factors = [.55, .62, .68, .72, .79, .83, .96, 1.01, 1.08, 1.24, 1.42, 2.05]
    rows = []
    for gamma in (1.0, 4.0):
        for series, multiplier in (("history", 1.0), ("qelm", 1.7), ("quadratic", .8)):
            for window in WINDOWS.astype(int):
                base = (.6 if gamma == 1.0 else .003) * multiplier / window
                for root, factor in enumerate(root_factors):
                    rows.append({
                        "root_id": root,
                        "teacher_gamma": gamma,
                        "window": window,
                        "nrmse": base * factor,
                        "series": series,
                        "metric": "history_nrmse" if series == "history" else "total_nrmse",
                        "teacher_n_qubits": 5,
                        "student_n_qubits": 10,
                        "qelm_feature_budget": 120,
                        "evidence_cohort": "SYNTHETIC_TEST_ONLY",
                    })
    # Deliberately unsorted. Plot coordinates must follow numeric window order.
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


@pytest.fixture
def synthetic_feature_source(synthetic_source) -> pd.DataFrame:
    """Three complete artificial budgets, paired with panel a/b at B=120."""
    baseline = synthetic_source.loc[
        synthetic_source["series"].eq("qelm") & synthetic_source["window"].eq(10),
        ["root_id", "teacher_gamma", "window", "nrmse",
         "teacher_n_qubits", "student_n_qubits"],
    ].rename(columns={"nrmse": "total_nrmse"})
    parts = []
    for budget, multiplier in zip(FEATURE_BUDGETS, (1.6, 1.2, 1.0), strict=True):
        frame = baseline.copy(deep=True)
        frame["feature_budget"] = budget
        frame["total_nrmse"] *= multiplier
        frame["snapshot_budget"] = budget // 30
        frame["evidence_cohort"] = "SYNTHETIC_TEST_ONLY"
        parts.append(frame)
    # Neither budget nor root order may determine the plotted aggregate order.
    return pd.concat(parts, ignore_index=True).iloc[::-1].reset_index(drop=True)


@pytest.fixture
def captured_render(tmp_path, monkeypatch, synthetic_source, synthetic_feature_source):
    """Capture artists; write only unmistakable test stubs, not figure media."""
    captured = []
    before = synthetic_source.copy(deep=True)
    features_before = synthetic_feature_source.copy(deep=True)

    def inspect(fig: Figure, filename, **kwargs):
        if not captured:
            fig.canvas.draw()
            captured.append(fig)
        else:
            assert captured[0] is fig
        Path(filename).write_bytes(b"SYNTHETIC TEST STUB: not a paper figure\n")

    monkeypatch.setattr(Figure, "savefig", inspect)
    paths = render(synthetic_source, tmp_path / "synthetic-test-only",
                   feature_source=synthetic_feature_source)
    pd.testing.assert_frame_equal(synthetic_source, before)
    pd.testing.assert_frame_equal(synthetic_feature_source, features_before)
    assert len(captured) == 1
    return captured[0], paths, synthetic_source, synthetic_feature_source


def test_all_long_context_panels_have_matching_major_grids_below_data(captured_render):
    fig, _, _, _ = captured_render
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


def test_three_square_panels_are_title_free_and_keep_the_original_window_scales(captured_render):
    fig, _, _, _ = captured_render
    assert len(fig.axes) == 3
    assert not fig.legends
    assert fig._suptitle is None
    for index, ax in enumerate(fig.axes):
        box = ax.get_position()
        assert box.width * fig.get_figwidth() == pytest.approx(
            box.height * fig.get_figheight()
        )
        assert 0 <= box.x0 < box.x1 <= 1
        assert 0 <= box.y0 < box.y1 <= 1
        assert all(ax.get_title(loc=loc) == "" for loc in ("left", "center", "right"))
        letters = [text for text in ax.texts if text.get_text() in {"a", "b", "c"}]
        assert [text.get_text() for text in letters] == ["abc"[index]]
        assert all(0 < value < 1 for value in letters[0].get_position())
        assert (ax.get_legend() is not None) == (index in (0, 2))
    for index, ax in enumerate(fig.axes[:2]):
        np.testing.assert_array_equal(ax.get_xticks(), WINDOWS)
        assert ax.get_yscale() == "log"
        formatter = ax.yaxis.get_major_formatter()
        assert formatter(.0001) == "0.01%"
        assert formatter(.001) == "0.1%"
        assert formatter(.1) == "10%"
        assert ax.get_xlabel() == r"window length $d$"
        assert ax.get_ylabel() == ("common-suffix error" if index == 0 else "")
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
    assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
    # These limits are the original two-panel renderer's synthetic-fixture limits.
    assert fig.axes[0].get_ylim() == pytest.approx((1e-4 / 1.2, 3.5751))
    assert fig.axes[0].get_xlim() == pytest.approx((1.6, 10.4))
    assert fig.axes[0].get_position().x1 < fig.axes[1].get_position().x0


def test_all_three_series_keep_all_five_twelve_root_medians_and_percentiles(captured_render):
    fig, _, source, _ = captured_render
    assert len(source) == 360
    assert source.groupby(["teacher_gamma", "series", "window"]).size().eq(12).all()
    for gamma, ax in zip((1.0, 4.0), fig.axes[:2], strict=True):
        bars = [item for item in ax.containers if isinstance(item, ErrorbarContainer)]
        assert len(bars) == 3
        assert [bar.get_label() for bar in bars] == [entry[1] for entry in SERIES]
        for (series, _, _, _, _), bar in zip(SERIES, bars, strict=True):
            grouped = source.loc[
                source["teacher_gamma"].eq(gamma) & source["series"].eq(series)
            ].groupby("window")["nrmse"]
            line, caps, whiskers = bar.lines
            np.testing.assert_array_equal(line.get_xdata(), WINDOWS)
            np.testing.assert_allclose(line.get_ydata(), grouped.median())
            assert len(caps) == 2
            assert len(whiskers) == 1
            assert isinstance(whiskers[0], LineCollection)
            lower = grouped.quantile(.1).to_numpy()
            upper = grouped.quantile(.9).to_numpy()
            expected = np.array([
                [[window, low], [window, high]]
                for window, low, high in zip(WINDOWS, lower, upper, strict=True)
            ])
            np.testing.assert_allclose(whiskers[0].get_segments(), expected)
            np.testing.assert_allclose(caps[0].get_ydata(), lower)
            np.testing.assert_allclose(caps[1].get_ydata(), upper)


def test_feature_panel_has_a_linear_percent_scale_and_discrete_budget_axis(captured_render):
    fig, _, _, _ = captured_render
    ax = fig.axes[2]
    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == "linear"
    assert ax.get_ylim()[0] == 0
    assert ax.get_ylim()[1] >= .4
    assert ax.get_xlabel() == r"measured features $B$"
    assert ax.get_ylabel() == "QELM error"
    np.testing.assert_array_equal(ax.get_xticks(), np.arange(3))
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["30", "60", "120"]
    formatter = ax.yaxis.get_major_formatter()
    assert formatter(0) == "0%"
    assert formatter(.1) == "10%"
    assert formatter(.4) == "40%"
    context = [text for text in ax.texts if text.get_text() == r"$d=10$"]
    assert len(context) == 1
    assert context[0].get_position() == pytest.approx((.95, .965))


def test_feature_panel_keeps_all_twelve_root_medians_and_asymmetric_percentiles(captured_render):
    fig, _, source, feature_source = captured_render
    assert len(feature_source) == 72
    assert feature_source.groupby(["teacher_gamma", "feature_budget"]).size().eq(12).all()
    assert feature_source["window"].eq(10).all()
    ax = fig.axes[2]
    bars = [item for item in ax.containers if isinstance(item, ErrorbarContainer)]
    assert len(bars) == 2
    for gamma, offset, bar in zip((1.0, 4.0), (-.09, .09), bars, strict=True):
        grouped = feature_source.loc[feature_source["teacher_gamma"].eq(gamma)].groupby(
            "feature_budget"
        )["total_nrmse"]
        positions = np.arange(3) + offset
        median = grouped.median().reindex(FEATURE_BUDGETS).to_numpy()
        lower = grouped.quantile(.1).reindex(FEATURE_BUDGETS).to_numpy()
        upper = grouped.quantile(.9).reindex(FEATURE_BUDGETS).to_numpy()
        line, caps, whiskers = bar.lines
        np.testing.assert_allclose(line.get_xdata(), positions)
        np.testing.assert_allclose(line.get_ydata(), median)
        assert len(caps) == 2
        assert len(whiskers) == 1
        assert isinstance(whiskers[0], LineCollection)
        expected = np.array([
            [[position, low], [position, high]]
            for position, low, high in zip(positions, lower, upper, strict=True)
        ])
        np.testing.assert_allclose(whiskers[0].get_segments(), expected)
        for cap, values in zip(caps, (lower, upper), strict=True):
            np.testing.assert_allclose(cap.get_xdata(), positions)
            np.testing.assert_allclose(cap.get_ydata(), values)
        assert not np.allclose(median - lower, upper - median)
        original = source.loc[
            source["teacher_gamma"].eq(gamma) & source["series"].eq("qelm") &
            source["window"].eq(10)
        ].set_index("root_id")["nrmse"].sort_index()
        repeated = feature_source.loc[
            feature_source["teacher_gamma"].eq(gamma) & feature_source["feature_budget"].eq(120)
        ].set_index("root_id")["total_nrmse"].sort_index()
        np.testing.assert_array_equal(repeated.to_numpy(), original.to_numpy())


def test_feature_panel_uses_coral_filled_squares_and_open_circles_without_connectors(captured_render):
    fig, _, _, _ = captured_render
    ax = fig.axes[2]
    bars = [item for item in ax.containers if isinstance(item, ErrorbarContainer)]
    assert [bar.get_label() for bar in bars] == [r"$\gamma_T=1$", r"$\gamma_T=4$"]
    for bar, marker, face in zip(bars, ("s", "o"), (COLORS["qelm"], "white"), strict=True):
        line, caps, whiskers = bar.lines
        assert line.get_marker() == marker
        assert line.get_linestyle().lower() == "none"
        assert to_rgba(line.get_color()) == to_rgba(COLORS["qelm"])
        assert to_rgba(line.get_markeredgecolor()) == to_rgba(COLORS["qelm"])
        assert to_rgba(line.get_markerfacecolor()) == to_rgba(face)
        for cap in caps:
            assert to_rgba(cap.get_color()) == to_rgba(COLORS["qelm"])
        np.testing.assert_allclose(whiskers[0].get_colors(), [to_rgba(COLORS["qelm"])])
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [r"$\gamma_T=1$", r"$\gamma_T=4$"]
    context = next(text for text in ax.texts if text.get_text() == r"$d=10$")
    assert legend.get_window_extent().y1 < context.get_window_extent().y0
    assert legend.get_window_extent().x1 > ax.get_window_extent().x0 + ax.get_window_extent().width / 2


def test_long_context_matches_resource_panel_size_and_typography_with_clear_gutters(captured_render):
    from matplotlib import pyplot as plt

    fig, _, _, _ = captured_render
    assert tuple(fig.get_size_inches()) == pytest.approx((7.055, 2.40))
    reference = plt.figure(figsize=tuple(fig.get_size_inches()))
    try:
        resource = _resource_diagnosis_axes(reference)
        resource_side = resource[0].get_position().width * reference.get_figwidth()
        assert resource_side == pytest.approx(COMPARISON_LAYOUT["panel_side"])
        for ax in fig.axes:
            position = ax.get_position()
            assert position.width * fig.get_figwidth() == pytest.approx(resource_side)
            assert position.height * fig.get_figheight() == pytest.approx(resource_side)
            assert position.y0 * fig.get_figheight() == pytest.approx(COMPARISON_LAYOUT["bottom"])
            assert ax.xaxis.label.get_fontsize() == STYLE["axes_labelsize"]
            assert ax.yaxis.label.get_fontsize() == STYLE["axes_labelsize"]
            assert ax.xaxis.labelpad == 4.0 and ax.yaxis.labelpad == 3.0
            assert all(tick.get_fontsize() == STYLE["tick_labelsize"] for tick in ax.get_yticklabels())
            letter = next(text for text in ax.texts if text.get_text() in {"a", "b", "c"})
            assert letter.get_position() == pytest.approx((.045, .965))
            assert letter.get_fontsize() == STYLE["axes_titlesize"]
        for left, right in zip(fig.axes[:-1], fig.axes[1:], strict=True):
            gap = (right.get_position().x0 - left.get_position().x1) * fig.get_figwidth()
            assert gap == pytest.approx(.68)
            labels = [*right.get_yticklabels()]
            if right.get_ylabel():
                labels.append(right.yaxis.label)
            gutter = min(text.get_window_extent().x0 for text in labels) - left.get_window_extent().x1
            assert gutter * 72 / fig.dpi >= 6.0
        for ax in (fig.axes[0], fig.axes[2]):
            legend = ax.get_legend()
            assert all(text.get_fontsize() == STYLE["legend_fontsize"] for text in legend.get_texts())
            outer, key = ax.get_window_extent(), legend.get_window_extent()
            assert outer.contains(key.x0, key.y0) and outer.contains(key.x1, key.y1)
    finally:
        plt.close(reference)


def test_no_faint_root_lines_or_bands_and_every_displayed_error_is_in_bounds(captured_render):
    fig, _, _, _ = captured_render
    for ax in fig.axes:
        assert not any(isinstance(artist, PolyCollection) for artist in ax.collections)
        bars = [item for item in ax.containers if isinstance(item, ErrorbarContainer)]
        members = [
            line for bar in bars for line in (bar.lines[0], *bar.lines[1])
        ]
        remaining = [line for line in ax.lines if line not in members]
        # One median marker/path and two cap paths per series; one 10% guide.
        assert len(ax.lines) == 3 * len(bars) + 1
        assert len(remaining) == 1
        np.testing.assert_allclose(remaining[0].get_ydata(), [.1, .1])
        assert all(line.get_alpha() in (None, 1) for line in ax.lines)
        assert all(artist.get_alpha() in (None, 1) for artist in ax.collections)
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        for bar in bars:
            line, _, collections = bar.lines
            assert np.all((np.asarray(line.get_ydata()) > ymin) &
                          (np.asarray(line.get_ydata()) < ymax))
            endpoints = np.asarray(collections[0].get_segments())
            assert np.all((endpoints[:, :, 0] > xmin) & (endpoints[:, :, 0] < xmax))
            assert np.all((endpoints[:, :, 1] > ymin) & (endpoints[:, :, 1] < ymax))


def test_reference_color_tokens_and_noncolor_series_cues(captured_render):
    fig, _, _, _ = captured_render
    assert {name: color for name, _, color, _, _ in SERIES} == COLORS
    for ax in fig.axes[:2]:
        bars = [item for item in ax.containers if isinstance(item, ErrorbarContainer)]
        for (series, _, color, marker, linestyle), bar in zip(SERIES, bars, strict=True):
            line = bar.lines[0]
            assert to_rgba(line.get_color()) == to_rgba(COLORS[series])
            assert line.get_marker() == marker
            assert line.get_linestyle() == linestyle
            assert line.get_linewidth() == pytest.approx(1.1)
            expected_face = "white" if series == "history" else color
            assert to_rgba(line.get_markerfacecolor()) == to_rgba(expected_face)
            for cap in bar.lines[1]:
                assert to_rgba(cap.get_color()) == to_rgba(COLORS[series])


def test_export_retains_every_synthetic_root_value_and_auditable_artifact_hashes(captured_render):
    _, (pdf, png, csv, features_csv), source, feature_source = captured_render
    assert csv.read_bytes() == source.to_csv(index=False).encode("utf-8")
    restored = pd.read_csv(csv)
    assert len(restored) == 360
    assert restored["evidence_cohort"].eq("SYNTHETIC_TEST_ONLY").all()
    pd.testing.assert_frame_equal(restored, source, check_exact=False, atol=1e-15)
    assert features_csv.read_bytes() == feature_source.to_csv(index=False).encode("utf-8")
    restored_features = pd.read_csv(features_csv)
    assert len(restored_features) == 72
    assert restored_features["evidence_cohort"].eq("SYNTHETIC_TEST_ONLY").all()
    pd.testing.assert_frame_equal(restored_features, feature_source, check_exact=False, atol=1e-15)
    manifest = json.loads((csv.parent / "figure_sha256.json").read_text())
    assert set(manifest) == {pdf.name, png.name, csv.name, features_csv.name}
    for path in (pdf, png, csv, features_csv):
        assert manifest[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert pdf.read_bytes().startswith(b"SYNTHETIC TEST STUB")
    assert png.read_bytes().startswith(b"SYNTHETIC TEST STUB")


@pytest.mark.parametrize("column", [
    "root_id", "teacher_gamma", "window", "feature_budget", "total_nrmse",
    "teacher_n_qubits", "student_n_qubits",
])
def test_feature_source_rejects_missing_columns(synthetic_source, synthetic_feature_source, column):
    with pytest.raises(ValueError):
        _validate_feature_source(synthetic_source, synthetic_feature_source.drop(columns=column))


@pytest.mark.parametrize("mutation", [
    "missing_row", "duplicate_row", "duplicate_key", "wrong_budget", "wrong_window",
    "wrong_gamma", "wrong_root", "foreign_cohort", "wrong_teacher_qubits",
    "wrong_student_qubits", "b120_mismatch",
])
def test_feature_source_rejects_incomplete_or_unpaired_cells(
    synthetic_source, synthetic_feature_source, mutation
):
    invalid = synthetic_feature_source.copy(deep=True)
    if mutation == "missing_row":
        invalid = invalid.iloc[1:]
    elif mutation == "duplicate_row":
        invalid = pd.concat([invalid, invalid.iloc[:1]], ignore_index=True)
    elif mutation == "duplicate_key":
        keys = ["root_id", "teacher_gamma", "window", "feature_budget"]
        invalid.loc[1, keys] = invalid.loc[0, keys]
    elif mutation == "foreign_cohort":
        invalid["root_id"] += 100
    elif mutation == "b120_mismatch":
        row = invalid.index[invalid["feature_budget"].eq(120)][0]
        invalid.loc[row, "total_nrmse"] *= 1.05
    else:
        column, value = {
            "wrong_budget": ("feature_budget", 90),
            "wrong_window": ("window", 8),
            "wrong_gamma": ("teacher_gamma", 2.0),
            "wrong_root": ("root_id", 99),
            "wrong_teacher_qubits": ("teacher_n_qubits", 6),
            "wrong_student_qubits": ("student_n_qubits", 5),
        }[mutation]
        invalid.loc[0, column] = value
    with pytest.raises(ValueError):
        _validate_feature_source(synthetic_source, invalid)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.1])
def test_feature_source_rejects_nonfinite_or_negative_errors(
    synthetic_source, synthetic_feature_source, value
):
    invalid = synthetic_feature_source.copy(deep=True)
    invalid.loc[0, "total_nrmse"] = value
    with pytest.raises(ValueError):
        _validate_feature_source(synthetic_source, invalid)


def test_render_rejects_invalid_feature_data_before_writing_artifacts(
    tmp_path, monkeypatch, synthetic_source, synthetic_feature_source
):
    def unexpected_save(*args, **kwargs):
        pytest.fail("Invalid source data reached figure export")

    monkeypatch.setattr(Figure, "savefig", unexpected_save)
    output = tmp_path / "invalid-source-must-not-export"
    with pytest.raises(ValueError):
        render(synthetic_source, output, feature_source=synthetic_feature_source.iloc[1:])
    assert not output.exists() or not any(output.iterdir())
