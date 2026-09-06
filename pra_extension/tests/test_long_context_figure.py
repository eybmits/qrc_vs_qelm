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

from render_long_context_n5 import SERIES, render  # noqa: E402


WINDOWS = np.array([2, 4, 6, 8, 10], dtype=float)
COLORS = {"history": "#000000", "qelm": "#00A14B", "quadratic": "#A93BFF"}


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
def captured_render(tmp_path, monkeypatch, synthetic_source):
    """Capture artists; write only unmistakable test stubs, not figure media."""
    captured = []
    before = synthetic_source.copy(deep=True)

    def inspect(fig: Figure, filename, **kwargs):
        if not captured:
            fig.canvas.draw()
            captured.append(fig)
        else:
            assert captured[0] is fig
        Path(filename).write_bytes(b"SYNTHETIC TEST STUB: not a paper figure\n")

    monkeypatch.setattr(Figure, "savefig", inspect)
    paths = render(synthetic_source, tmp_path / "synthetic-test-only")
    pd.testing.assert_frame_equal(synthetic_source, before)
    assert len(captured) == 1
    return captured[0], paths, synthetic_source


def test_two_square_panels_are_title_free_and_share_window_and_error_scales(captured_render):
    fig, _, _ = captured_render
    assert len(fig.axes) == 2
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
        np.testing.assert_array_equal(ax.get_xticks(), WINDOWS)
        assert ax.get_yscale() == "log"
        formatter = ax.yaxis.get_major_formatter()
        assert formatter(.0001) == "0.01%"
        assert formatter(.001) == "0.1%"
        assert formatter(.1) == "10%"
        assert ax.get_xlabel() == r"window length $d$"
        assert ax.get_ylabel() == "common-suffix error"
        letters = [text for text in ax.texts if text.get_text() in {"a", "b"}]
        assert [text.get_text() for text in letters] == ["ab"[index]]
        assert all(0 < value < 1 for value in letters[0].get_position())
        assert (ax.get_legend() is not None) == (index == 0)
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
    assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
    assert fig.axes[0].get_position().x1 < fig.axes[1].get_position().x0


def test_all_three_series_keep_all_five_twelve_root_medians_and_percentiles(captured_render):
    fig, _, source = captured_render
    assert len(source) == 360
    assert source.groupby(["teacher_gamma", "series", "window"]).size().eq(12).all()
    for gamma, ax in zip((1.0, 4.0), fig.axes, strict=True):
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


def test_no_faint_root_lines_or_bands_and_every_displayed_error_is_in_bounds(captured_render):
    fig, _, _ = captured_render
    for ax in fig.axes:
        assert not any(isinstance(artist, PolyCollection) for artist in ax.collections)
        bars = [item for item in ax.containers if isinstance(item, ErrorbarContainer)]
        members = [
            line for bar in bars for line in (bar.lines[0], *bar.lines[1])
        ]
        remaining = [line for line in ax.lines if line not in members]
        # Three median paths, six cap paths, and one 10% reference only.
        assert len(ax.lines) == 10
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
    fig, _, _ = captured_render
    assert {name: color for name, _, color, _, _ in SERIES} == COLORS
    for ax in fig.axes:
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
    _, (pdf, png, csv), source = captured_render
    assert csv.read_text() == source.to_csv(index=False)
    restored = pd.read_csv(csv)
    assert len(restored) == 360
    assert restored["evidence_cohort"].eq("SYNTHETIC_TEST_ONLY").all()
    pd.testing.assert_frame_equal(restored, source, check_exact=False, atol=1e-15)
    manifest = json.loads((csv.parent / "figure_sha256.json").read_text())
    assert set(manifest) == {pdf.name, png.name, csv.name}
    for path in (pdf, png, csv):
        assert manifest[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert pdf.read_bytes().startswith(b"SYNTHETIC TEST STUB")
    assert png.read_bytes().startswith(b"SYNTHETIC TEST STUB")
