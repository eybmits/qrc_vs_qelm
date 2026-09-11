"""Three-column layout: unchanged traces, distinct cohorts, readable geometry."""
import json
from pathlib import Path
import sys

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
import numpy as np
from pypdf import PdfReader

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import render_memory_prediction as renderer  # noqa: E402


def test_three_panel_source_binding_and_all_controls():
    original, task = renderer.load_sources()
    with renderer.plt.rc_context():
        fig, p = renderer.make_figure(original, task)
        assert len(fig.axes) == 9
        assert [t.get_text() for t in fig.texts if t.get_fontweight() == "bold"] == ["a", "b", "c"]
        np.testing.assert_array_equal(p["laser"][0].lines[0].get_ydata(), original["laser_input"])
        for ax, rate in zip(p["laser"][1:], (4., .2)):
            for line, model in zip(ax.lines, ("qrc", "qelm")):
                np.testing.assert_array_equal(line.get_ydata(), original["laser_outputs"][str(rate)][model])
                np.testing.assert_array_equal(line.get_xdata(), np.arange(128))
        for model in ("qrc", "qelm"):
            ax = p["narma"][model]
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), task["traces"]["narma"]["target"])
            np.testing.assert_array_equal(ax.lines[1].get_ydata(), task["traces"]["narma"][model])
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), np.arange(96))
        assert p["narma"]["qrc"].get_ylim() == p["narma"]["qelm"].get_ylim()
        rows = [r for r in task["all_rows"] if r["study"] == "narma" and r["model"] == "replay"]
        windows = sorted({r["window"] for r in rows})
        medians = [np.median([r["nrmse"] for r in rows if r["window"] == d]) for d in windows]
        np.testing.assert_array_equal(p["narma"]["history"].lines[0].get_xdata(), windows)
        np.testing.assert_array_equal(p["narma"]["history"].lines[0].get_ydata(), medians)
        assert all(sum(r["window"] == d for r in rows) == 24 for d in windows)
        classical = next(c for c in p["narma"]["history"].containers if c.get_label() == "Quadratic")
        np.testing.assert_array_equal(classical.lines[0].get_xdata(), [4, 12])
        for rate in (4., .2):
            values = np.asarray(original["paired_outputs"][str(rate)])
            line = next(item for item in p["memory"].lines if item.get_gid() == f"memory_gap_gamma{rate:g}")
            np.testing.assert_array_equal(line.get_ydata(), np.median(abs(values[:,:,0]-values[:,:,1]), axis=0)[:48])
        exact = original["convergence_rows"]
        np.testing.assert_array_equal(p["exact"].lines[0].get_ydata(), [r["error"] for r in exact])
        np.testing.assert_array_equal(p["exact"].lines[1].get_ydata(), [exact[0]["floor"]]*2)
        assert p["exact"].lines[1].get_color() == "#E53935"
        renderer.plt.close(fig)


def test_three_panel_geometry_and_legend_clearance():
    with renderer.plt.rc_context():
        fig, p = renderer.make_figure(*renderer.load_sources())
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        draw = canvas.get_renderer()
        for ax in fig.axes:
            box = ax.get_tightbbox(draw)
            assert box.x0 >= 0 and box.y0 >= 0
            assert box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        assert min(t.get_fontsize() for t in texts) >= 7.5
        for key in ("narma_legend", "model_legend"):
            box = p[key].get_window_extent(draw)
            assert all(not box.overlaps(ax.get_tightbbox(draw)) for ax in fig.axes)
        annotation = p["memory"].texts[0].get_window_extent(draw)
        assert not annotation.overlaps(p["memory_legend"].get_window_extent(draw))
        labels = [t.get_window_extent(draw) for t in p["history_inset"].texts]
        assert not labels[0].overlaps(labels[1])
        for ax, legend in ((p["exact"], p["exact_legend"]), (p["memory"], p["memory_legend"])):
            box = legend.get_window_extent(draw)
            for line in ax.lines:
                if not line.get_gid() and ax is p["memory"]:
                    continue
                points = np.column_stack((line.get_xdata(), line.get_ydata()))
                assert not any(box.contains(*point) for point in ax.transData.transform(points))
        renderer.plt.close(fig)


def test_three_panel_reproducible_export_and_source_retention(tmp_path):
    first = renderer.render(tmp_path / "first")
    second = renderer.render(tmp_path / "second")
    assert first == second
    assert first["emulation"] == json.loads(renderer.EMULATION.read_text())
    assert first["narma_trace"]["seed"] == 921000
    assert first["narma_trace"]["test_indices"] == list(range(128, 224))
    assert len(first["narma_all_rows"]) == 264
    assert len(first["narma_candidate_rows"]) == 192
    for name, expected in first["artifacts"].items():
        assert renderer.digest(tmp_path / "first" / name) == expected
    pdf = PdfReader(tmp_path / "first" / f"{renderer.STEM}.pdf")
    assert len(pdf.pages) == 1
    np.testing.assert_allclose([float(pdf.pages[0].mediabox.width), float(pdf.pages[0].mediabox.height)],
                               np.array(renderer.SIZE)*72)
