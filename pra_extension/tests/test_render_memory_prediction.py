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
        assert [t.get_text() for t in fig.texts] == ["a", "b", "c"]
        assert all(t.get_fontweight() == "bold" for t in fig.texts)
        assert all(not ax.get_title() for ax in fig.axes)
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
            # The same model uses the same solid trace and marker treatment in
            # measured-input emulation and the separate forecasting example.
            reference = p["laser"][1].lines[0 if model == "qrc" else 1]
            for attribute in ("linestyle", "linewidth", "marker", "markersize",
                              "markerfacecolor", "markeredgewidth"):
                assert getattr(ax.lines[1], "get_" + attribute)() == (
                    getattr(reference, "get_" + attribute)()
                )
            assert ax.lines[1].get_markevery() == (1 if model == "qrc" else 3)
            assert ax.lines[0].get_linewidth() == ax.lines[1].get_linewidth()
            assert ax.lines[0].get_alpha() == 1.
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
            np.testing.assert_array_equal(line.get_xdata(), np.arange(1, 49))
        # Zoom only the viewport; the complete 48-input trajectory remains bound
        # to the source data above, including the hidden continuation.
        assert p["memory"].get_xlim() == (.4, 32.6)
        np.testing.assert_array_equal(p["memory"].get_xticks(), [1, 8, 16, 24, 32])
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


@pytest.mark.parametrize("font_family", [None, "DejaVu Sans"],
                         ids=["preferred-font", "bundled-fallback-font"])
def test_three_panel_geometry_and_legend_clearance(font_family):
    with renderer.plt.rc_context():
        fig, p = renderer.make_figure(*renderer.load_sources())
        if font_family is not None:
            # Exercise the bundled Linux fallback even on hosts with Helvetica.
            for text in fig.findobj(Text):
                text.set_fontfamily(font_family)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        draw = canvas.get_renderer()
        for ax in fig.axes:
            box = ax.get_tightbbox(draw)
            assert box.x0 >= 0 and box.y0 >= 0
            assert box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        assert min(t.get_fontsize() for t in texts) >= 7.5
        # Shared frame edges keep the three columns readable after removing
        # descriptive headings and compacting the space above the upper group.
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
        # These labels align at the top. Their glyph-dependent heights differ
        # with DejaVu Sans, so equal bottom edges are not a layout invariant.
        assert all(ax.xaxis.label.get_verticalalignment() == "top" for ax in lower_row)
        np.testing.assert_allclose([box.y1 for box in label_boxes], label_boxes[0].y1)
        for ax, lower in ((p["memory"], p["exact"]),
                          (p["narma"]["qelm"], p["narma"]["history"])):
            label = ax.xaxis.label.get_window_extent(draw)
            assert label.y0 - lower.bbox.y1 >= draw.points_to_pixels(6)
        # Every lower legend stays fully inside its own axes with a small inset.
        for ax, key in zip(lower_row,
                ("model_legend", "exact_legend", "narma_legend")):
            legend = p[key]
            box = legend.get_window_extent(draw)
            assert legend.axes is ax and legend.get_frame_on()
            assert legend.get_frame().get_alpha() == 1.
            assert ax.bbox.y0 < box.y0 < box.y1 < ax.bbox.y1
            assert ax.bbox.x0 < box.x0 < box.x1 < ax.bbox.x1
            assert not box.overlaps(ax.xaxis.label.get_window_extent(draw))
        # Laser condition/error labels now use clear areas inside their axes.
        for ax, condition in zip(p["laser"][1:], ("Fast loss", "Slow loss")):
            assert p["condition_labels"][condition] in ax.texts
            assert len(ax.texts) == 2
            boxes = [text.get_window_extent(draw) for text in ax.texts]
            assert not boxes[0].overlaps(boxes[1])
            for box in boxes:
                assert ax.bbox.x0 < box.x0 < box.x1 < ax.bbox.x1
                assert ax.bbox.y0 < box.y0 < box.y1 < ax.bbox.y1
                if ax is p["laser"][-1]:
                    assert not box.overlaps(p["model_legend"].get_window_extent(draw))
                for line in ax.lines:
                    path = line.get_path().transformed(line.get_transform())
                    assert not path.intersects_bbox(box.padded(draw.points_to_pixels(1)), filled=False)
        # Test line segments, not only vertices: the old in-axes labels crossed
        # target segments even when no sampled point lay inside their boxes.
        for model in ("qrc", "qelm"):
            ax = p["narma"][model]
            legend = p["narma_trace_legends"][model]
            assert [text.get_text() for text in legend.texts] == ["Target", model.upper()]
            assert legend.axes is ax
            boxes = [legend.get_window_extent(draw)]
            for box in boxes:
                assert box.y0 > ax.bbox.y1
                assert box.x0 >= ax.bbox.x0 and box.x1 <= ax.bbox.x1
                assert box.y1 < (fig.texts[2].get_window_extent(draw).y0
                                if model == "qrc" else p["narma"]["qrc"].bbox.y0)
                for line in ax.lines:
                    path = line.get_path().transformed(line.get_transform())
                    assert not path.intersects_bbox(box, filled=False)
        annotation = p["memory"].texts[0].get_window_extent(draw)
        assert not annotation.overlaps(p["memory_legend"].get_window_extent(draw))
        labels = [t.get_window_extent(draw) for t in p["history_inset"].texts]
        assert not labels[0].overlaps(labels[1])
        for ax, legend in ((p["laser"][-1], p["model_legend"]),
                           (p["exact"], p["exact_legend"]), (p["memory"], p["memory_legend"]),
                           (p["narma"]["history"], p["narma_legend"])):
            box = legend.get_window_extent(draw)
            for line in ax.lines:
                if not line.get_gid() and ax is p["memory"]:
                    continue
                # The vertical ten-input guide may continue behind the box.
                if ax is p["narma"]["history"] and line.get_transform() is ax.get_xaxis_transform():
                    continue
                points = np.column_stack((line.get_xdata(), line.get_ydata())).astype(float)
                assert not any(box.contains(*point) for point in line.get_transform().transform(points))
                path = line.get_path().transformed(line.get_transform())
                assert not path.intersects_bbox(box.padded(draw.points_to_pixels(1)), filled=False)
        history = p["narma"]["history"]
        assert history.get_ylim() == (.4, 1.05)
        assert [t.get_text() for t in p["narma_legend"].texts] == [
            "QRC: last d", "QRC: full", "QELM: last 4"]
        legend_box = p["narma_legend"].get_window_extent(draw)
        for band in history.collections:
            for path in band.get_paths():
                assert not path.transformed(band.get_transform()).intersects_bbox(legend_box, filled=True)
        guide_label = history.texts[0].get_window_extent(draw)
        assert not guide_label.overlaps(p["narma_legend"].get_window_extent(draw))
        for line in history.lines[:2]:
            assert not line.get_path().transformed(line.get_transform()).intersects_bbox(guide_label, filled=False)
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
    assert first["memory_gap_display"] == {
        "first_input": 1, "last_input": 32, "source_inputs_retained": 48,
    }
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
