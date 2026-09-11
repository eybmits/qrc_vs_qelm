"""Binding/geometry guards for the author-selected Figure 3 design."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pypdf import PdfReader

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import render_figure3_candidate as candidate  # noqa: E402


@pytest.fixture
def source():
    return pd.read_csv(CODE.parent.parent /
                       "paper/figures/long_context_n5/fig04_long_context_n5_source.csv")


@pytest.fixture
def inputs(source):
    # Unit-test arrays only. Published output always reads the predeclared data.
    x = np.arange(128)
    raw = (128 + 100 * np.sin(x / 10)).astype(float)
    series = {4.: (.9 + .05 * np.cos(x / 10), .9 + .049 * np.cos(x / 10)),
              .2: (.5 + .2 * np.cos(x / 13), .55 + .15 * np.cos(x / 13))}
    errors = {4.: .031, .2: .305}
    exact = candidate.convergence_rows(d=8, eta=.8, beta=2., copies=candidate.COPIES)
    paired = {}
    for rate, decay in ((4., 4.), (.2, 24.)):
        paired[rate] = np.array([np.column_stack((.5 + gap, np.full(128, .5),
                                                  .5 + gap/2, .5 + gap/2))
                                 for gap in [(.03 + j*.008)*np.exp(-x/decay) for j in range(12)]])
    histories = np.array([np.r_[.5+.4*np.sin(x/10), .5+.4*np.sin(x/13)],
                          np.r_[.5+.4*np.cos(x/10), .5+.4*np.sin(x/13)]])
    return raw, series, errors, source, exact, paired, histories


def test_preserves_values_rates_and_separate_metrics(inputs):
    with candidate.plt.rc_context(candidate.chart_style()):
        fig, panels = candidate.build_figure(*inputs)
    assert len(fig.axes) == 6
    raw, series, _, source, exact, paired, histories = inputs
    np.testing.assert_array_equal(panels["laser"][0].lines[0].get_ydata(), raw)
    for ax, rate in zip(panels["laser"][1:], candidate.Config().rates):
        for line, expected in zip(ax.lines, series[rate]):
            np.testing.assert_array_equal(line.get_ydata(), expected)
            np.testing.assert_array_equal(line.get_xdata(), np.arange(128))
    lines = {line.get_gid(): line for line in panels["memory"].lines if line.get_gid()}
    assert len(lines) == 2
    for rate, values in paired.items():
        line = lines[f"memory_gap_gamma{rate:g}"]
        median = np.median(abs(values[:,:,0] - values[:,:,1]), axis=0)
        np.testing.assert_array_equal(line.get_xdata(), np.arange(1, 49))
        np.testing.assert_array_equal(line.get_ydata(), median[:48])
    assert panels["memory"].get_yscale() == "linear"
    assert panels["memory"].get_ylabel() == "QRC output gap"
    assert [t.get_text() for t in panels["history_inset"].texts] == [
        "Different earlier inputs", "Same inputs from here on"]
    assert [t.get_text() for t in panels["memory_legend"].get_texts()] == [
        "Fast loss", "Slow loss"]
    for i in range(2):
        baseline = .18 + (1-i)*.48
        np.testing.assert_array_equal(panels["history_inset"].lines[2*i].get_ydata(),
                                      baseline + .31*histories[i,:128])
    assert panels["exact"].get_yscale() == "linear"
    assert "absolute error" in panels["exact"].get_ylabel()
    np.testing.assert_array_equal(panels["exact"].lines[0].get_xdata(), candidate.COPIES)
    np.testing.assert_array_equal(panels["exact"].lines[0].get_ydata(), [r["error"] for r in exact])
    np.testing.assert_array_equal(panels["exact"].lines[1].get_ydata(), [exact[0]["floor"]] * 2)
    floor = panels["exact"].lines[1]
    assert floor.get_gid() == "missing_history_floor"
    assert floor.get_color() == candidate.HISTORY_FLOOR_RED == "#E53935"
    assert floor.is_dashed()
    assert floor.get_label() == "Unavoidable window error"
    assert "Unavoidable window error" in [
        text.get_text() for text in panels["exact"].get_legend().get_texts()
    ]
    candidate.plt.close(fig)


def test_physical_dimensions_labels_and_legends(inputs, tmp_path):
    hashes = []
    for index in range(2):
        output = tmp_path / str(index)
        output.mkdir()
        with candidate.plt.rc_context(candidate.chart_style()):
            fig, panels = candidate.build_figure(*inputs)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            for ax in fig.axes:
                box = ax.get_tightbbox(renderer)
                assert box.x0 >= 0 and box.y0 >= 0
                assert box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1
            window_label = panels["memory"].xaxis.label.get_window_extent(renderer)
            heading = next(t for t in fig.texts if t.get_text() == "Approaching the exact limit")
            assert not window_label.overlaps(heading.get_window_extent(renderer))
            for ax, legends in ((panels["memory"], (panels["memory_legend"],)),
                                (panels["exact"], (panels["exact"].get_legend(),))):
                for legend in legends:
                    box = legend.get_window_extent(renderer)
                    for line in ax.lines:
                        if line.get_gid() or ax is panels["exact"]:
                            points = np.column_stack((line.get_xdata(), line.get_ydata()))
                            assert not any(box.contains(*p) for p in ax.transData.transform(points))
                    for collection in ax.collections:
                        for path in collection.get_paths():
                            assert not any(box.contains(*p) for p in ax.transData.transform(path.vertices))
            history_box = panels["history_inset"].get_window_extent(renderer)
            label_boxes = [t.get_window_extent(renderer) for t in panels["history_inset"].texts]
            assert not label_boxes[0].overlaps(label_boxes[1])
            for box in label_boxes:
                assert box.x0 > history_box.x0 and box.x1 < history_box.x1
                assert box.y0 > history_box.y0 and box.y1 < history_box.y1
            memory_box = panels["memory"].get_window_extent(renderer)
            assert history_box.overlaps(memory_box)
            for line in panels["memory"].lines:
                if line.get_gid():
                    points = np.column_stack((line.get_xdata(), line.get_ydata()))
                    assert not any(history_box.contains(*p) for p in panels["memory"].transData.transform(points))
            hashes.append(candidate.save(fig, output, "candidate"))
    assert hashes[0] == hashes[1]
    page = PdfReader(tmp_path / "0/candidate.pdf").pages[0]
    assert float(page.mediabox.width) == pytest.approx(6.221 * 72)
    assert float(page.mediabox.height) == pytest.approx(candidate.SIZE[1] * 72)


def test_rejects_missing_observations(inputs):
    args = list(inputs)
    args[3] = args[3].iloc[:-1]
    with pytest.raises(ValueError, match="all 360"):
        candidate.build_figure(*args)
    args = list(inputs)
    args[5] = dict(args[5])
    args[5][4.] = args[5][4.][:-1]
    with pytest.raises(ValueError, match="all 12 seeds"):
        candidate.build_figure(*args)


def test_caption_preserves_scope_and_normalization():
    assert "training-output SD" in candidate.CAPTION
    assert "not laser forecasting" in candidate.CAPTION
    assert "constructed, not learned" in candidate.CAPTION
    assert "absolute, not NRMSE" in candidate.CAPTION
    assert "12 roots per rate" in candidate.CAPTION
    assert "pair-specific gap, not the global optimum" in candidate.CAPTION
    assert "slow loss is not permanent memory" in candidate.CAPTION
    assert "red dashed line" in candidate.CAPTION
    assert "unavoidable error from missing history" in candidate.CAPTION


def test_selected_paper_asset_retains_cohorts_and_all_seed_statistics(source):
    directory = CODE.parent.parent / "paper/figures/emulation_evidence"
    ledger = json.loads((directory / "fig04_emulation_evidence_source.json").read_text())
    assert ledger["status"] == "author_selected_publication_asset"
    assert ledger["figure_size_inches"] == list(candidate.SIZE)
    for filename, expected in ledger["artifacts"].items():
        assert candidate.digest(directory / filename) == expected
    for filename, expected in ledger["sources"].items():
        assert candidate.digest(CODE.parent.parent / filename) == expected
    recorded = pd.DataFrame(ledger["window_rows"])
    columns = ["root_id", "teacher_gamma", "series", "window"]
    pd.testing.assert_frame_equal(recorded[source.columns].sort_values(columns).reset_index(drop=True),
                                  source.sort_values(columns).reset_index(drop=True),
                                  check_exact=False, rtol=1e-13, atol=1e-15)
    assert ledger["convergence_rows"] == candidate.convergence_rows(
        d=8, eta=.8, beta=2., copies=candidate.COPIES)
    assert ledger["paired_seed_order"] == list(candidate.Config().seeds)
    assert ledger["paired_display_steps"] == [1, 48]
    paired = {float(k): np.array(v) for k, v in ledger["paired_outputs"].items()}
    for rate, summary in candidate.memory_statistics(paired).items():
        for key, values in summary.items():
            np.testing.assert_allclose(values, ledger["memory_statistics"][str(rate)][key], atol=1e-15)
    summary = ledger["laser_all_seed_summary"]
    assert all(summary["checks"].values())
    assert len(summary["results"]) == 24
    assert {(r["seed"], r["gamma"]) for r in summary["results"]} == {
        (seed, rate) for seed in candidate.Config().seeds for rate in candidate.Config().rates}
    for rate in candidate.Config().rates:
        cells = [r for r in summary["results"] if r["gamma"] == rate]
        values = [r["test_nrmse_training_sd"] for r in cells]
        group = summary["groups"][str(rate)]
        assert group["n_seeds"] == 12
        assert group["nrmse_median"] == pytest.approx(np.median(values))
        assert group["nrmse_min_max"] == [min(values), max(values)]
        assert all(r["n_features_each"] == 48 and r["test_scored_rows"] == 1920 for r in cells)
        assert all(r["pair"]["max_shared_prediction_difference"] == 0 for r in cells)
        shown = next(r for r in cells if r["seed"] == candidate.Config().seeds[0])
        assert ledger["laser_outputs"][str(rate)]["full_test_nrmse_training_sd"] == shown["test_nrmse_training_sd"]
    specification = ledger["laser_run_specification"]
    assert candidate.digest(CODE / "laser_demonstration.py") == specification["runner_sha256"]
    assert candidate.digest(CODE.parent / "LASER_DEMONSTRATION_PROTOCOL.md") == specification["protocol_sha256"]


def test_restored_emulation_figure_preserves_additional_task_evidence():
    """Restore the emulation figure without dropping the new task evidence."""
    base = CODE.parent.parent / "paper/icaart2027"
    evaluation = " ".join((base / "sections/evaluation.tex").read_text().split())
    assert "fig04_memory_prediction.pdf" in evaluation
    assert "fig04_task_memory.pdf" not in evaluation
    assert "NARMA10" in evaluation and "forecast" in evaluation.lower()
    assert "weigend1993timeseries" in evaluation
    bibliography = (base / "references.bib").read_text()
    assert "weigend1993timeseries" in bibliography
    assert "fujii2017harnessing" in bibliography
    archived = base.parent / "figures/task_memory"
    for suffix in (".pdf", ".png", "_source.json", "_rows.csv", "_traces.csv"):
        assert (archived / f"fig04_task_memory{suffix}").is_file()
    protocol = " ".join((CODE.parent / "LASER_DEMONSTRATION_PROTOCOL.md").read_text().split())
    for text in ("QRC-output emulation, not prediction of future laser intensity",
                 "48 exact local X/Y/Z expectations", "917000 through 917011",
                 "no refit on validation or test", "global D_d/2 and not an RMSE bound"):
        assert text in protocol
