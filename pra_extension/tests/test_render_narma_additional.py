"""The optional NARMA asset preserves the fixed fresh evidence and legibility."""
import hashlib
from pathlib import Path
import sys

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
import numpy as np
from pypdf import PdfReader

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import render_narma_additional as renderer  # noqa: E402


def test_optional_narma_preserves_fresh_source_and_renders_deterministically(tmp_path):
    parent = renderer.load_source()
    first = renderer.render(output=tmp_path / "first")
    second = renderer.render(output=tmp_path / "second")
    assert first == second
    assert first["status"] == "optional_addition_not_active_manuscript"
    assert first["trace"] == parent["traces"]["narma"]
    assert first["trace"]["seed"] == 921000
    assert first["trace"]["test_indices"] == list(range(128, 224))
    assert first["candidate_rows"] == parent["candidate_rows"]
    assert len(first["root_rows"]) == 264
    assert len(first["candidate_rows"]) == 192
    assert {r["seed"] for r in first["root_rows"]} == set(range(921000, 921024))
    assert all(r["n"] == 24 for r in first["aggregates"])
    assert first["verification"]["verification_passed"]
    for name, digest in first["artifacts"].items():
        data = (tmp_path / "first" / name).read_bytes()
        assert data == (tmp_path / "second" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
    pdf = PdfReader(tmp_path / "first/narma_additional.pdf")
    np.testing.assert_allclose([float(pdf.pages[0].mediabox.width), float(pdf.pages[0].mediabox.height)],
                               np.array([6.221, 2.4])*72)


def test_optional_narma_direct_trace_binding_controls_and_layout():
    parent = renderer.load_source()
    with renderer.plt.rc_context():
        fig, axes, legends = renderer.make_figure(parent)
        assert len(axes) == 3
        trace = parent["traces"]["narma"]
        for model in ("qrc", "qelm"):
            np.testing.assert_array_equal(axes[model].lines[0].get_ydata(), trace["target"])
            np.testing.assert_array_equal(axes[model].lines[1].get_ydata(), trace[model])
            np.testing.assert_array_equal(axes[model].lines[0].get_xdata(), np.arange(96))
        assert axes["qrc"].get_ylim() == axes["qelm"].get_ylim()
        labels = axes["history"].get_legend_handles_labels()[1]
        assert set(labels) == {"replay/refit", "QRC (full)", "QELM(4)", "quadratic"}
        classical = next(c for c in axes["history"].containers if c.get_label() == "quadratic")
        np.testing.assert_array_equal(classical.lines[0].get_xdata(), [4, 12])
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        draw = canvas.get_renderer()
        for ax in axes.values():
            bbox = ax.get_tightbbox(draw)
            assert bbox.x0 >= 0 and bbox.y0 >= 0 and bbox.x1 <= fig.bbox.x1 and bbox.y1 <= fig.bbox.y1
        for legend in legends:
            bbox = legend.get_window_extent(draw)
            assert all(not bbox.overlaps(ax.get_tightbbox(draw)) for ax in axes.values())
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        assert min(t.get_fontsize() for t in texts) >= 7.5
        assert "NARMA10 (synthetic)" in [t.get_text() for t in texts]
        renderer.plt.close(fig)
