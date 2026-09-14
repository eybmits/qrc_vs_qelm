"""Three-column layout: unchanged traces, distinct cohorts, readable geometry."""
import json
from copy import deepcopy
from pathlib import Path
import sys

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
import numpy as np
import pytest
from pypdf import PdfReader

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import render_memory_prediction as renderer  # noqa: E402


def test_dense_curve_rejects_unverified_or_stale_report():
    _, _, dense = renderer.load_sources()
    missing = deepcopy(dense)
    missing["verification"].pop("independent")
    with pytest.raises(ValueError, match="requires independent"):
        renderer.require_verified_dense(missing)
    stale = deepcopy(dense)
    stale["verification"]["independent"]["checkpoint_sha256"]["seed921000.npz"] = "0" * 64
    with pytest.raises(ValueError, match="not bound"):
        renderer.require_verified_dense(stale)


def test_three_panel_source_binding_and_selected_controls():
    original, task, dense = renderer.load_sources()
    with renderer.plt.rc_context():
        fig, p = renderer.make_figure(original, task, dense)
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
            assert ax.lines[0].get_color() == renderer.roles.TARGET
            assert ax.lines[1].get_color() == getattr(renderer.roles, model.upper())
        for ax in p["laser"][1:]:
            assert [line.get_color() for line in ax.lines] == [renderer.roles.QRC, renderer.roles.QELM]
        assert p["laser"][0].lines[0].get_color() == renderer.roles.INPUT
        assert p["laser"][0].get_position().x0 < p["memory"].get_position().x0 < p["narma"]["qrc"].get_position().x0
        assert p["narma"]["qrc"].get_ylim() == p["narma"]["qelm"].get_ylim()
        rows = dense["rows"]
        windows = sorted({r["window"] for r in rows})
        assert windows == list(range(1, 49))
        medians = [np.median([r["nrmse"] for r in rows if r["window"] == d]) for d in windows]
        np.testing.assert_array_equal(p["narma"]["history"].lines[0].get_xdata(), windows)
        np.testing.assert_array_equal(p["narma"]["history"].lines[0].get_ydata(), medians)
        assert all(sum(r["window"] == d for r in rows) == 24 for d in windows)
        history = p["narma"]["history"]
        assert not any("Quadratic" in c.get_label() for c in history.containers)
        assert [line.get_color() for line in history.lines[:2]] == [renderer.roles.QRC, renderer.FULL_HISTORY_COLOR]
        assert history.get_xscale() == "linear"
        qelm = next(c for c in history.containers if c.get_label() == "QELM: last 4 inputs")
        np.testing.assert_array_equal(qelm.lines[0].get_xdata(), [4])
        assert qelm.lines[0].get_color() == renderer.roles.QELM
        # The dip below continuous QRC at d=12 is measured, not smoothed away.
        assert medians[windows.index(12)] < history.lines[1].get_ydata()[0]
        for rate in (4., .2):
            values = np.asarray(original["paired_outputs"][str(rate)])
            line = next(item for item in p["memory"].lines if item.get_gid() == f"memory_gap_gamma{rate:g}")
            np.testing.assert_array_equal(line.get_ydata(), np.median(abs(values[:,:,0]-values[:,:,1]), axis=0)[:48])
            assert line.get_color() == renderer.MEMORY_COLORS[rate]
        for line in p["history_inset"].lines[:4]:
            earlier = np.max(line.get_xdata()) < 128
            assert line.get_color() == (renderer.EARLIER_INPUT_COLOR if earlier else renderer.roles.INPUT)
        assert [text.get_color() for text in p["history_inset"].texts] == [
            renderer.EARLIER_INPUT_COLOR, renderer.roles.INPUT]
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
        # Shared frame edges and separate text rows keep the three columns readable.
        np.testing.assert_allclose([ax.bbox.width for ax in fig.axes], fig.axes[0].bbox.width)
        first_row = [p["laser"][0], p["history_inset"], p["narma"]["qrc"]]
        upper_bottom = [p["laser"][1], p["memory"], p["narma"]["qelm"]]
        lower_row = [p["laser"][-1], p["exact"], p["narma"]["history"]]
        np.testing.assert_allclose([ax.bbox.y1 for ax in first_row], first_row[0].bbox.y1)
        np.testing.assert_allclose([ax.bbox.y0 for ax in upper_bottom], upper_bottom[0].bbox.y0)
        for edge in ("y0", "y1"):
            np.testing.assert_allclose([getattr(ax.bbox, edge) for ax in lower_row],
                                       getattr(lower_row[0].bbox, edge))
        label_boxes = [ax.xaxis.label.get_window_extent(draw) for ax in lower_row]
        np.testing.assert_allclose([box.y0 for box in label_boxes], label_boxes[0].y0)
        for ax, heading in ((p["memory"], "exact"), (p["narma"]["qelm"], "history")):
            label = ax.xaxis.label.get_window_extent(draw)
            title = p["section_headings"][heading].get_window_extent(draw)
            assert label.y0 - title.y1 >= draw.points_to_pixels(6)
        footers = [p[key].get_window_extent(draw)
                   for key in ("model_legend", "exact_legend", "narma_legend")]
        np.testing.assert_allclose([box.y1 for box in footers], footers[0].y1)
        for ax, label, footer in zip(lower_row, label_boxes, footers):
            assert footer.y1 < label.y0 < ax.bbox.y0
        # Test line segments, not only vertices: the old in-axes labels crossed
        # target segments even when no sampled point lay inside their boxes.
        for model in ("qrc", "qelm"):
            ax = p["narma"][model]
            boxes = [text.get_window_extent(draw) for text in ax.texts]
            if model == "qrc":
                boxes.append(p["narma_target_legend"].get_window_extent(draw))
            for box in boxes:
                assert box.y0 > ax.bbox.y1
                for line in ax.lines:
                    path = line.get_path().transformed(line.get_transform())
                    assert not path.intersects_bbox(box, filled=False)
        for key in ("model_legend",):
            box = p[key].get_window_extent(draw)
            assert all(not box.overlaps(ax.get_tightbbox(draw)) for ax in fig.axes)
        annotation = p["memory"].texts[0].get_window_extent(draw)
        assert not annotation.overlaps(p["memory_legend"].get_window_extent(draw))
        labels = [t.get_window_extent(draw) for t in p["history_inset"].texts]
        assert not labels[0].overlaps(labels[1])
        for ax, legend in ((p["exact"], p["exact_legend"]), (p["memory"], p["memory_legend"]),
                           (p["narma"]["history"], p["narma_legend"])):
            box = legend.get_window_extent(draw)
            for line in ax.lines:
                if not line.get_gid() and ax is p["memory"]:
                    continue
                points = np.column_stack((line.get_xdata(), line.get_ydata())).astype(float)
                assert not any(box.contains(*point) for point in line.get_transform().transform(points))
        renderer.plt.close(fig)


def test_three_panel_reproducible_export_and_source_retention(tmp_path):
    first = renderer.render(tmp_path / "first")
    second = renderer.render(tmp_path / "second")
    assert first == second
    assert first["emulation"] == json.loads(renderer.EMULATION.read_text())
    assert first["narma_trace"]["seed"] == 921000
    assert first["narma_trace"]["test_indices"] == list(range(128, 224))
    assert len(first["narma_all_rows"]) == 264
    assert len(first["narma_dense_replay"]["rows"]) == 24 * 48
    assert first["narma_dense_replay"]["verification"]["independent"]["passed"]
    assert len(first["narma_candidate_rows"]) == 192
    assert {r["window"] for r in first["narma_all_rows"] if r["model"] == "quadratic"} == {4, 12}
    assert "24-root" in first["panels"]["c"]
    assert "12-root" in first["panels"]["b"]
    for name, expected in first["artifacts"].items():
        assert renderer.digest(tmp_path / "first" / name) == expected
    pdf = PdfReader(tmp_path / "first" / f"{renderer.STEM}.pdf")
    assert len(pdf.pages) == 1
    np.testing.assert_allclose([float(pdf.pages[0].mediabox.width), float(pdf.pages[0].mediabox.height)],
                               np.array(renderer.SIZE)*72)
