"""Check the integrated figure without changing or pooling its source evidence."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pypdf import PdfReader
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import render_long_context_sharpness as renderer  # noqa: E402


@pytest.fixture(scope="module")
def sources():
    original = CODE.parent.parent / "paper/figures/long_context_n5"
    return (
        pd.read_csv(original / "fig04_long_context_n5_source.csv"),
        pd.read_csv(original / "fig04_long_context_n5_features_source.csv"),
    )


def test_combined_figure_preserves_all_rows_and_is_deterministic(tmp_path, sources):
    source, features = sources
    first = renderer.render(source, tmp_path / "first", feature_source=features)
    second = renderer.render(source, tmp_path / "second", feature_source=features)
    assert first == second
    assert first["window_rows"] == source.to_dict(orient="records")
    assert first["feature_rows"] == features.to_dict(orient="records")
    assert len(first["window_rows"]) == 360
    assert len(first["feature_rows"]) == 72
    assert first["convergence_rows"] == renderer.convergence_rows(
        d=8, eta=.8, beta=2.0, copies=renderer.COPIES
    )
    for name, digest in first["artifacts"].items():
        data = (tmp_path / "first" / name).read_bytes()
        assert data == (tmp_path / "second" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
    ledger = tmp_path / "first" / f"{renderer.STEM}_source.json"
    assert json.loads(ledger.read_text()) == first
    pdf = PdfReader(tmp_path / "first" / f"{renderer.STEM}.pdf")
    assert len(pdf.pages) == 1
    assert float(pdf.pages[0].mediabox.width) == pytest.approx(7.055 * 72)
    assert float(pdf.pages[0].mediabox.height) == pytest.approx(3.95 * 72)


def test_panels_keep_separate_metrics_and_fit_the_publication_canvas(tmp_path, sources, monkeypatch):
    captured = []
    close = renderer.plt.close

    def capture_and_close(fig):
        if hasattr(fig, "axes"):
            captured.append(fig)
        close(fig)

    monkeypatch.setattr(renderer.plt, "close", capture_and_close)
    ledger = renderer.render(sources[0], tmp_path, feature_source=sources[1])
    fig = captured[-1]
    assert len(fig.axes) == 4
    a, b, c, d = fig.axes
    assert a.get_yscale() == b.get_yscale() == "log"
    assert c.get_yscale() == d.get_yscale() == "linear"
    assert d.get_xscale() == "log"
    assert all("%" in ax.yaxis.get_major_formatter()(.1, 0) for ax in (a, b, c))
    assert "%" not in d.yaxis.get_major_formatter()(.1, 0)
    assert "absolute error" in d.get_ylabel()
    assert "constructed" in [text.get_text() for text in d.texts]
    assert [text.get_text() for text in d.get_legend().get_texts()] == [
        "Constructed QELM", "Best possible error"
    ]
    np.testing.assert_array_equal(d.lines[0].get_xdata(), renderer.COPIES)
    np.testing.assert_array_equal(d.lines[0].get_ydata(),
                                  [row["error"] for row in ledger["convergence_rows"]])
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    canvas_renderer = canvas.get_renderer()
    for ax in fig.axes:
        bbox = ax.get_tightbbox(canvas_renderer)
        assert bbox.x0 >= 0 and bbox.y0 >= 0
        assert bbox.x1 <= fig.bbox.x1 and bbox.y1 <= fig.bbox.y1
    # The feature legend must not hide the original root-percentile whiskers.
    legend_box = c.get_legend().get_window_extent(canvas_renderer)
    for collection in c.collections:
        for segment in collection.get_segments():
            for point in c.transData.transform(segment):
                assert not legend_box.contains(*point)


def test_combined_figure_rejects_dropped_feature_observations(tmp_path, sources):
    with pytest.raises(ValueError, match="72 unique"):
        renderer.render(sources[0], tmp_path, feature_source=sources[1].iloc[:-1])
