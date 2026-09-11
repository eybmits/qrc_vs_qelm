"""Source fidelity and publication layout for the fresh task-memory figure."""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
import numpy as np
from pypdf import PdfReader
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import render_task_memory as renderer  # noqa: E402


@pytest.fixture(scope="module")
def payload():
    path = renderer.DEFAULT_OUTPUT / f"{renderer.STEM}_source.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_ledger_preserves_all_fresh_conditions_candidates_and_source_identities(payload):
    renderer.validate_payload(payload)
    assert len(payload["all_rows"]) == 1704
    assert len(payload["candidate_rows"]) == 192
    assert len(payload["aggregates"]) == 71
    assert all(row["n"] == 24 for row in payload["aggregates"])
    for study in ("laser", "narma"):
        expected = []
        for cell in sorted(payload["study_summaries"][study]["results"],
                           key=lambda c: (c["seed"], c.get("gamma", 0))):
            condition = ("fast" if cell["gamma"] == 4 else "slow") if study == "laser" else "selected"
            for row in cell["rows"]:
                expected.append(dict(row) | {"study": study, "condition": condition,
                    "seed": cell["seed"], "horizon": row.get("horizon", 1),
                    "gamma": cell.get("gamma", row.get("gamma"))})
        assert [row for row in payload["all_rows"] if row["study"] == study] == expected
    expected_candidates = [dict(row) | {"seed": cell["seed"]}
                           for cell in sorted(payload["study_summaries"]["narma"]["results"], key=lambda c: c["seed"])
                           for row in cell["candidate_rows"]]
    assert payload["candidate_rows"] == expected_candidates
    # Outcomes not drawn in the small main panels are still retained in full.
    laser = [row for row in payload["all_rows"] if row["study"] == "laser"]
    assert {row["horizon"] for row in laser} == {1, 4}
    assert {row["gamma"] for row in laser} == {.2, 4}
    assert {row["model"] for row in laser} == {"qrc", "qelm", "replay_refit", "replay_frozen", "persistence"}
    assert {row["window"] for row in payload["all_rows"] if row["model"] == "quadratic"} == {4, 12}
    assert "pra_extension/code/render_task_memory.py" in payload["sources"]
    assert "pra_extension/code/task_memory_replication.py" in payload["sources"]
    assert "pra_extension/TASK_MEMORY_REPLICATION_PROTOCOL.md" in payload["sources"]
    assert "pra_extension/code/paper_palette.py" in payload["sources"]
    for path, digest in payload["sources"].items():
        assert hashlib.sha256((renderer.ROOT / path).read_bytes()).hexdigest() == digest, path
    assert len(payload["caption"].split()) <= 120
    assert "not confidence intervals" in payload["scope"]["interval"]
    assert "no past targets supplied" in payload["scope"]["narma"]
    assert "more evolution steps" in payload["scope"]["history"]
    assert "no quantum advantage" in payload["scope"]["claims"]
    verification = payload["independent_verification"]
    assert verification["verification_passed"] is True
    assert verification["direction_checks"] == payload["direction_checks"]
    assert sum(study["score_rows_recomputed"] for study in verification["studies"].values()) == 1896
    assert any(path.endswith("/verification.json") for path in payload["data_sources"])


def test_percentiles_are_recomputed_from_each_complete_root_score_group(payload):
    for group in payload["aggregates"]:
        keys = ("study", "condition", "model", "window", "horizon")
        rows = [row for row in payload["all_rows"] if all(row[k] == group[k] for k in keys)]
        assert len(rows) == len({row["seed"] for row in rows}) == 24
        expected = np.percentile([row["nrmse"] for row in rows], [10, 50, 90])
        np.testing.assert_array_equal(expected, [group["p10"], group["median"], group["p90"]])
    assert set(payload["direction_checks"]) == {
        "narma_qelm_minus_qrc", "narma_replay4_minus_qrc",
        "laser_fast_h1_qelm_minus_qrc", "laser_fast_h1_replay4_minus_qrc"}
    for gate in payload["direction_checks"].values():
        assert gate["total_roots"] == 24
        assert gate["passed"] == (gate["positive_roots"] >= 20 and gate["mean_ci95"][0] > 0)


def test_all_display_traces_are_direct_unsmoothed_first_root_arrays(payload):
    with renderer.plt.rc_context():
        fig, axes, _ = renderer.make_figure(payload)
        assert len(axes) == 8
        for study, seed in (("laser", 922000), ("narma", 921000)):
            trace = payload["traces"][study]
            assert trace["seed"] == seed
            assert trace["test_indices"] == list(range(128, 224))
            assert trace["target_indices"] == list(range(129, 225))
            np.testing.assert_array_equal(axes[f"{study}_input"].lines[0].get_ydata(), trace["input"])
        for condition in ("fast", "slow"):
            ax = axes[f"laser_{condition}"]
            assert len(ax.lines) == 3
            for line, model in zip(ax.lines, ("target", "qrc", "qelm"), strict=True):
                expected = payload["traces"]["laser"]["target"] if model == "target" else payload["traces"]["laser"]["conditions"][condition][model]
                np.testing.assert_array_equal(line.get_xdata(), np.arange(96))
                np.testing.assert_array_equal(line.get_ydata(), expected)
        for model in ("qrc", "qelm"):
            ax = axes[f"narma_{model}"]
            assert len(ax.lines) == 2
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), payload["traces"]["narma"]["target"])
            np.testing.assert_array_equal(ax.lines[1].get_ydata(), payload["traces"]["narma"][model])
        assert axes["laser_fast"].get_ylim() == axes["laser_slow"].get_ylim()
        assert axes["narma_qrc"].get_ylim() == axes["narma_qelm"].get_ylim()
        assert axes["narma_input"].get_ylim() == (0, .2)
        assert axes["laser_fast"].lines[0].get_color() == renderer.colors.COBALT
        assert axes["laser_fast"].lines[1].get_color() == renderer.colors.INK
        assert axes["laser_fast"].lines[2].get_color() == renderer.colors.CORAL
        for study in ("laser", "narma"):
            ax = axes[f"{study}_history"]
            assert ax.get_xscale() == "log"
            assert ax.get_ylim()[0] == 0
            assert ax.yaxis.get_major_formatter()(.1, 0) == "10%"
            names = ax.get_legend_handles_labels()[1]
            conditions = ("fast", "slow") if study == "laser" else ("selected",)
            assert all(f"replay_{c}" in names and f"qrc_{c}" in names and f"qelm_{c}" in names for c in conditions)
            if study == "narma":
                assert "quadratic" in names
        renderer.plt.close(fig)


def test_publication_canvas_legends_fonts_and_labels_have_clearance(payload):
    with renderer.plt.rc_context():
        fig, axes, legends = renderer.make_figure(payload)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        draw = canvas.get_renderer()
        assert tuple(fig.get_size_inches()) == (6.221, 3.65)
        for ax in axes.values():
            bbox = ax.get_tightbbox(draw)
            assert bbox.x0 >= 0 and bbox.y0 >= 0, (ax.get_label(), bbox)
            assert bbox.x1 <= fig.bbox.x1 and bbox.y1 <= fig.bbox.y1, (ax.get_label(), bbox)
            assert any(line.get_visible() for line in ax.get_xgridlines())
            assert any(line.get_visible() for line in ax.get_ygridlines())
        for legend in legends:
            box = legend.get_window_extent(draw)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1
            assert all(not box.overlaps(ax.get_tightbbox(draw)) for ax in axes.values())
        texts = [a for a in fig.findobj(Text) if a.get_visible() and a.get_text()]
        assert all(a.get_fontsize() >= 7.5 for a in texts)
        assert {"a", "b", "c"} <= {a.get_text() for a in texts}
        assert not any("floor" in a.get_text().lower() or "sharpness" in a.get_text().lower() for a in texts)
        renderer.plt.close(fig)


def test_exports_are_deterministic_and_keep_complete_csv_rows(tmp_path, payload):
    first = renderer.render_payload(payload, tmp_path / "one")
    second = renderer.render_payload(payload, tmp_path / "two")
    assert first == second
    for name, digest in first["artifacts"].items():
        data = (tmp_path / "one" / name).read_bytes()
        assert data == (tmp_path / "two" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
    for suffix, count in (("_rows.csv", 1704), ("_traces.csv", 192)):
        with (tmp_path / "one" / f"{renderer.STEM}{suffix}").open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == count
    pdf = PdfReader(tmp_path / "one" / f"{renderer.STEM}.pdf")
    assert len(pdf.pages) == 1
    assert float(pdf.pages[0].mediabox.width) == pytest.approx(6.221 * 72)
    assert float(pdf.pages[0].mediabox.height) == pytest.approx(3.65 * 72)


@pytest.mark.parametrize("change", ("drop_root", "duplicate_root", "drop_rate", "drop_horizon", "drop_candidate",
                                    "wrong_seed", "wrong_interval", "wrong_alignment", "nan_trace", "wrong_aggregate"))
def test_incomplete_or_edited_display_sources_are_rejected(payload, change):
    damaged = deepcopy(payload)
    if change == "drop_root":
        damaged["all_rows"].pop()
    elif change == "duplicate_root":
        damaged["all_rows"].append(damaged["all_rows"][0])
    elif change == "drop_rate":
        damaged["all_rows"] = [r for r in damaged["all_rows"] if r["condition"] != "slow"]
    elif change == "drop_horizon":
        damaged["all_rows"] = [r for r in damaged["all_rows"] if r["horizon"] != 4]
    elif change == "drop_candidate":
        damaged["candidate_rows"].pop()
    elif change == "wrong_seed":
        damaged["traces"]["narma"]["seed"] += 1
    elif change == "wrong_interval":
        damaged["traces"]["laser"]["test_indices"] = list(range(129, 225))
    elif change == "wrong_alignment":
        damaged["traces"]["laser"]["target_indices"] = list(range(128, 224))
    elif change == "nan_trace":
        damaged["traces"]["laser"]["conditions"]["slow"]["qelm"][0] = float("nan")
    else:
        damaged["aggregates"][0]["median"] += .01
    with pytest.raises(ValueError):
        renderer.validate_payload(damaged)
