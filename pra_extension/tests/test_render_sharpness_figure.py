"""Keep the new sharpness panel deterministic and distinct from fitted errors."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pypdf import PdfReader
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import render_sharpness_figure as renderer  # noqa: E402
from sharpness_benchmarks import convergence_rows  # noqa: E402


def test_render_uses_exact_source_rows_and_deterministic_artifacts(tmp_path):
    first = renderer.render(tmp_path / "first")
    second = renderer.render(tmp_path / "second")
    assert first == second
    assert first["rows"] == convergence_rows(
        d=8, eta=0.8, beta=2.0, copies=renderer.COPIES
    )
    assert first["rows"][0]["floor"] == pytest.approx(0.08388608, abs=1e-15)
    assert first["metric"] == "global worst-case absolute scalar error; not NRMSE"
    for name, expected_hash in first["artifacts"].items():
        first_bytes = (tmp_path / "first" / name).read_bytes()
        assert first_bytes == (tmp_path / "second" / name).read_bytes()
        assert hashlib.sha256(first_bytes).hexdigest() == expected_hash
    source = tmp_path / "first" / f"{renderer.STEM}_source.json"
    assert json.loads(source.read_text()) == first
    pdf = PdfReader(tmp_path / "first" / f"{renderer.STEM}.pdf")
    assert len(pdf.pages) == 1
    box = pdf.pages[0].mediabox
    assert float(box.width) == pytest.approx(renderer.FIGURE_SIZE[0] * 72)
    assert float(box.height) == pytest.approx(renderer.FIGURE_SIZE[1] * 72)


def test_axes_and_labels_do_not_claim_optimal_capacity_or_percentages(tmp_path, monkeypatch):
    figures = []
    close = renderer.plt.close

    def capture_and_close(figure):
        if hasattr(figure, "axes"):
            figures.append(figure)
        close(figure)

    monkeypatch.setattr(renderer.plt, "close", capture_and_close)
    report = renderer.render(tmp_path)
    ax = figures[-1].axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "linear"
    assert ax.get_xlabel() == r"Encoding copies per input $r$"
    assert ax.get_ylabel() == "Worst-case absolute error"
    assert ax.get_title() == ""
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Constructed QELM", "Best possible error"
    ]
    curve, floor = ax.lines
    assert tuple(curve.get_xdata()) == renderer.COPIES
    np.testing.assert_array_equal(curve.get_ydata(), [row["error"] for row in report["rows"]])
    np.testing.assert_array_equal(floor.get_ydata(), [report["rows"][0]["floor"]] * 2)
    assert curve.get_color() == renderer.colors.COBALT
    assert floor.get_color() == renderer.colors.INK
    assert curve.get_marker() == "o"
    assert curve.get_linestyle() != floor.get_linestyle()
    assert all(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())


def test_renderer_rejects_incomplete_or_invalid_source_rows(tmp_path, monkeypatch):
    rows = convergence_rows(d=8, eta=0.8, beta=2.0, copies=renderer.COPIES)
    monkeypatch.setattr(renderer, "convergence_rows", lambda **_: rows[:-1])
    with pytest.raises(ValueError, match="copy-count coverage"):
        renderer.render(tmp_path)
    rows[0]["error"] = rows[0]["floor"] - 0.01
    monkeypatch.setattr(renderer, "convergence_rows", lambda **_: rows)
    with pytest.raises(ValueError, match="information floor"):
        renderer.render(tmp_path)
