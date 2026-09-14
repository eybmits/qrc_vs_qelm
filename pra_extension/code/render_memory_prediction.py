"""Three-column Figure 3 from existing, separately verified evidence only.

No fitting, simulation, smoothing, resampling or excerpt selection occurs here.
The earlier two-column figure and NARMA-only figure remain recoverable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import paper_palette as colors
import active_paper_palette as roles
from render_figure3_candidate import build_figure, chart_style
from render_narma_additional import load_source as load_task_source, digest
from render_task_memory import _summary
from narma_dense_replay import load_source as load_dense_replay

ROOT = Path(__file__).resolve().parents[2]
EMULATION = ROOT / "paper/figures/emulation_evidence/fig04_emulation_evidence_source.json"
TASK = ROOT / "paper/figures/task_memory/fig04_task_memory_source.json"
DENSE_REPLAY = ROOT / "paper/figures/narma_dense_replay/source.json"
OUTPUT = ROOT / "paper/figures/memory_prediction"
STEM = "fig04_memory_prediction"
FULL_HISTORY_COLOR = "#C43B3B"
EARLIER_INPUT_COLOR = "#C43B3B"
MEMORY_COLORS = {4.: roles.QRC, .2: colors.VIOLET}
NARMA_TARGET_ALPHA = 1.
NARMA_DISPLAY_STEPS = 48
SIZE = (6.221, 4.47)
COL_LEFT = {"a": .47, "b": 2.53, "c": 4.59}
PLOT_WIDTH = 1.51
UPPER_TOP, UPPER_BOTTOM = 4.09, 2.30
LOWER_TOP, LOWER_BOTTOM = 1.35, .57
PANEL_TITLE_Y, SECTION_TITLE_Y = 4.29, 1.75
NARMA_TRACE_HEIGHT = .78


def require_verified_dense(dense):
    report = dense["verification"].get("independent", {})
    if not report.get("passed", False):
        raise ValueError("Dense history curve requires independent verification")
    checkpoints = report["checkpoint_sha256"]
    artifacts = {Path(name).name: value for name, value in dense["result_artifacts"].items()}
    seeds = sorted({row["seed"] for row in dense["rows"]})
    expected = {"extension_specification.json"} | {
        f"seed{seed}.{ext}" for seed in seeds for ext in ("json", "npz")}
    if (set(checkpoints) != expected
            or any(artifacts.get(name) != value for name, value in checkpoints.items())
            or checkpoints["extension_specification.json"] != dense["specification_sha256"]
            or report["parent_sha256"] != dense["parent_sha256"]
            or report["sources"] != dense["sources"]
            or report["verifier_sha256"] != digest(Path(__file__).with_name("verify_narma_dense_replay.py"))):
        raise ValueError("Independent verification is not bound to this dense history curve")


def load_sources():
    emulation = json.loads(EMULATION.read_text())
    for relative, expected in emulation["sources"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Changed emulation dependency: {relative}")
    for name, expected in emulation["artifacts"].items():
        if digest(EMULATION.parent / name) != expected:
            raise ValueError(f"Changed emulation artifact: {name}")
    task = load_task_source(TASK)
    dense = load_dense_replay(DENSE_REPLAY)
    require_verified_dense(dense)
    return emulation, task, dense


def _position(ax, left, bottom, width, height):
    ax.set_position([left / SIZE[0], bottom / SIZE[1], width / SIZE[0], height / SIZE[1]])


def _axis(fig, left, bottom, width, height, name):
    ax = fig.add_axes([left / SIZE[0], bottom / SIZE[1], width / SIZE[0], height / SIZE[1]], label=name)
    ax.set_axisbelow(True)
    ax.grid(color=colors.GRID, lw=.4, ls=(0, (4, 4)))
    ax.tick_params(length=2.3, width=.6, pad=2)
    return ax


def _attached_legend(ax, handles, labels, *, ncol=1, anchor_y=1.40):
    """Attach the legend to the top-right frame, with room above the data."""
    legend = ax.legend(handles, labels, loc="upper right",
        bbox_to_anchor=(1., anchor_y), frameon=True, fancybox=True,
        facecolor="white", edgecolor=roles.TARGET, framealpha=1., fontsize=7.5,
        handlelength=1.2, handletextpad=.4, labelspacing=.15, borderpad=.35,
        borderaxespad=0, ncol=ncol, columnspacing=.8)
    legend.get_frame().set_linewidth(.6)
    return legend


def make_figure(emulation, task, dense):
    plt.rcParams.update(chart_style())
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8,
                        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    outputs = emulation["laser_outputs"]
    fig, panels = build_figure(
        np.asarray(emulation["laser_input"]),
        {float(r): (v["qrc"], v["qelm"]) for r, v in outputs.items()},
        {float(r): v["full_test_nrmse_training_sd"] for r, v in outputs.items()},
        pd.DataFrame(emulation["window_rows"]), emulation["convergence_rows"],
        {float(r): np.asarray(v) for r, v in emulation["paired_outputs"].items()},
        emulation["paired_history_inputs"],
    )
    fig.set_size_inches(*SIZE, forward=True)
    for text in list(fig.texts):
        text.remove()
    for legend in list(fig.legends):
        legend.remove()
    for letter, title in (("a", "Laser-driven emulation"),
                          ("b", "Memory and exact limit"),
                          ("c", "NARMA10 prediction")):
        x = COL_LEFT[letter]
        fig.text(x / SIZE[0], PANEL_TITLE_Y / SIZE[1], letter, fontsize=9, weight="bold")
        fig.text((x + .15) / SIZE[0], PANEL_TITLE_Y / SIZE[1], title, fontsize=8)

    # a: Keep every one of the original 128 excerpt samples.
    for ax, bottom, height in zip(panels["laser"],
        (UPPER_TOP - NARMA_TRACE_HEIGHT, UPPER_BOTTOM, LOWER_BOTTOM),
        (NARMA_TRACE_HEIGHT, NARMA_TRACE_HEIGHT, LOWER_TOP - LOWER_BOTTOM), strict=True):
        _position(ax, COL_LEFT["a"], bottom, PLOT_WIDTH, height)
        ax.set_xticks([0, 64, 127])
        ax.set_ylabel(ax.get_ylabel(), labelpad=3)
    panels["laser"][-1].set_xlabel("Test step (excerpt)", labelpad=3)
    panels["laser"][0].lines[0].set_color(roles.INPUT)
    panels["laser"][0].tick_params(axis="x", bottom=False)
    section_headings = {}
    for ax, label, title_y in zip(panels["laser"][1:],
        ("Fast loss", "Slow loss"), (3.16, SECTION_TITLE_Y), strict=True):
        for line, color in zip(ax.lines, (roles.QRC, roles.QELM), strict=True):
            line.set_color(color)
        old_label, error_label = ax.texts
        old_label.remove()
        section_headings[label] = fig.text(COL_LEFT["a"] / SIZE[0], title_y / SIZE[1],
            label, fontsize=7.5)
        error_label.set_fontsize(7.5)
        error_label.set_text(error_label.get_text().replace("full test:", "test:"))
        error_label.set_transform(fig.transFigure)
        error_label.set_position(((COL_LEFT["a"] + PLOT_WIDTH) / SIZE[0], title_y / SIZE[1]))
        error_label.set_verticalalignment("baseline")
    handles, labels = panels["laser"][1].get_legend_handles_labels()
    model_legend = _attached_legend(panels["laser"][-1], handles, labels,
                                    ncol=2, anchor_y=1.08)

    # b: The memory control follows the measured-input emulation in a.
    _position(panels["memory"], COL_LEFT["b"], UPPER_BOTTOM, PLOT_WIDTH, NARMA_TRACE_HEIGHT)
    panels["memory"].set_xticks([1, 16, 32, 48])
    panels["memory"].set_xlabel("Shared inputs received", labelpad=3)
    panels["memory"].set_ylabel("QRC output gap", labelpad=3)
    panels["memory"].texts[0].set_text("Same window from input 4")
    panels["memory"].texts[0].set_fontsize(7.5)
    panels["memory"].texts[0].set_y(.30)
    for line in panels["memory"].lines:
        if line.get_gid():
            rate = float(line.get_gid().removeprefix("memory_gap_gamma"))
            line.set_color(MEMORY_COLORS[rate])
    for band, rate in zip(panels["memory"].collections, (4., .2), strict=True):
        band.set_facecolor(MEMORY_COLORS[rate])
    panels["memory"].get_legend().remove()
    panels["memory_legend"] = panels["memory"].legend(loc="upper right", ncol=2,
        frameon=False, fontsize=7.5, handlelength=.9, handletextpad=.3,
        columnspacing=.8, borderpad=.25)
    _position(panels["history_inset"], COL_LEFT["b"], UPPER_TOP - NARMA_TRACE_HEIGHT,
              PLOT_WIDTH, NARMA_TRACE_HEIGHT)
    panels["history_inset"].set_ylim(.1, 1.9)
    for line in panels["history_inset"].lines[:4]:
        line.set_color(EARLIER_INPUT_COLOR if np.max(line.get_xdata()) < 128 else roles.INPUT)
    for text, words, color in zip(panels["history_inset"].texts,
        ("Different earlier\ninputs", "Same inputs\nfrom here on"),
        (EARLIER_INPUT_COLOR, roles.INPUT), strict=True):
        text.set_text(words)
        text.set_fontsize(7.5)
        text.set_color(color)
    _position(panels["exact"], COL_LEFT["b"], LOWER_BOTTOM, PLOT_WIDTH, LOWER_TOP - LOWER_BOTTOM)
    panels["exact"].lines[0].set_color(roles.QELM)
    panels["exact"].set_xticks([1, 16, 256], labels=[1, 16, 256])
    panels["exact"].set_xlabel("Copies per input", labelpad=3)
    panels["exact"].set_ylabel("Worst-case\nabsolute error", labelpad=3)
    panels["exact"].get_legend().remove()
    exact_handles, _ = panels["exact"].get_legend_handles_labels()
    exact_legend = _attached_legend(panels["exact"], exact_handles,
        ["Constructed QELM", "Unavoidable\nwindow error"])
    section_headings["exact"] = fig.text(COL_LEFT["b"] / SIZE[0], SECTION_TITLE_Y / SIZE[1],
        "Analytical benchmark", fontsize=7.5)

    # c: Zoom into the fixed excerpt's first 48 samples; retain all 96 source rows.
    trace = task["traces"]["narma"]
    values = np.asarray([trace[k] for k in ("target", "qrc", "qelm")])
    lo, hi = values.min(), values.max()
    narma = {}
    for model, bottom in (("qrc", UPPER_TOP - NARMA_TRACE_HEIGHT), ("qelm", UPPER_BOTTOM)):
        label_y = bottom + NARMA_TRACE_HEIGHT + .025
        ax = _axis(fig, COL_LEFT["c"], bottom, PLOT_WIDTH, NARMA_TRACE_HEIGHT, "narma_" + model)
        narma[model] = ax
        ax.plot(np.arange(96), trace["target"], color=roles.TARGET,
            alpha=NARMA_TARGET_ALPHA, lw=.75, label="Target")
        ax.plot(np.arange(96), trace[model], color=roles.QRC if model == "qrc" else roles.QELM,
            lw=.7, ls="--" if model == "qrc" else "-", marker=None if model == "qrc" else "o",
            ms=2.0, mfc="white", mew=.5, markevery=8, label=model.upper())
        ax.set_ylim(lo - .08 * (hi-lo), hi + .30 * (hi-lo))
        ax.set_yticks([.18, .21, .24])
        ax.set_xlim(0, NARMA_DISPLAY_STEPS - 1)
        ax.set_xticks([0, NARMA_DISPLAY_STEPS // 2, NARMA_DISPLAY_STEPS - 1])
        ax.set_ylabel("Output", labelpad=3)
        ax.text((COL_LEFT["c"] + PLOT_WIDTH) / SIZE[0], label_y / SIZE[1], model.upper(),
            ha="right", va="bottom", fontsize=7.5, transform=fig.transFigure)
        if model == "qrc":
            ax.tick_params(labelbottom=False, bottom=False)
            target_legend = ax.legend(handles=[ax.lines[0]], loc="lower left",
                bbox_to_anchor=(COL_LEFT["c"] / SIZE[0], label_y / SIZE[1]),
                bbox_transform=fig.transFigure, frameon=False, fontsize=7.5,
                handlelength=1.0, handletextpad=.3, borderpad=0, borderaxespad=0)
        else:
            ax.set_xlabel("Test step (excerpt)", labelpad=3)
    section_headings["history"] = fig.text(COL_LEFT["c"] / SIZE[0], SECTION_TITLE_Y / SIZE[1],
        "How much history is needed?", fontsize=7.5)
    history = _axis(fig, COL_LEFT["c"], LOWER_BOTTOM, PLOT_WIDTH, LOWER_TOP - LOWER_BOTTOM, "narma_history")
    narma["history"] = history
    replay = sorted(dense["aggregates"], key=lambda row: row["window"])
    windows = np.array([r["window"] for r in replay])
    med, low, high = [np.array([r[k] for r in replay]) for k in ("median", "p10", "p90")]
    history.fill_between(windows, low, high, color=roles.QRC, alpha=.15, lw=0)
    history.plot(windows, med, color=roles.QRC, marker="o", ms=2.5, mew=0, lw=.9,
        label="QRC: last d inputs")
    qrc = _summary(task, "narma", "selected", "qrc")[0]
    history.axhspan(qrc["p10"], qrc["p90"], color=FULL_HISTORY_COLOR, alpha=.06, lw=0)
    history.axhline(qrc["median"], color=FULL_HISTORY_COLOR, lw=.95, ls=(0, (4, 3)),
        label="QRC: full history")
    qelm = _summary(task, "narma", "selected", "qelm")[0]
    history.errorbar([4], [qelm["median"]],
        yerr=[[qelm["median"]-qelm["p10"]], [qelm["p90"]-qelm["median"]]],
        color=roles.QELM, marker="o", mfc="white", ms=3.5, lw=.8, capsize=1.5,
        label="QELM: last 4 inputs", zorder=5)
    # The independent quadratic comparison remains in prose and all source rows.
    history.set_xlim(.5, 20.5)
    history.set_xticks([1, 5, 10, 15, 20])
    history.set_ylim(.40, .90)
    history.set_yticks([.4, .6, .8])
    history.axvline(10, color=colors.MUTED, lw=.6, ls=(0, (2, 3)), zorder=0)
    history.text(10.7, .45, "10 inputs", fontsize=7.5, color=roles.TARGET, va="center")
    history.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:g}%"))
    history.set_xlabel("Recent inputs available, d", labelpad=3)
    history.set_ylabel("Test error", labelpad=3)
    narma_handles = [
        Line2D([], [], color=roles.QRC, marker="o", ms=3.1, lw=1, label="QRC: last d inputs"),
        Line2D([], [], color=FULL_HISTORY_COLOR, lw=.95, ls=(0, (4, 3)), label="QRC: full history"),
        Line2D([], [], color=roles.QELM, marker="o", mfc="white", ms=3.5, lw=0,
            label="QELM: last 4 inputs")]
    narma_legend = _attached_legend(history, narma_handles,
        [handle.get_label() for handle in narma_handles])
    for ax in fig.axes:
        ax.tick_params(length=2.3, width=.6, pad=2)
    panels.update(narma=narma, narma_legend=narma_legend, exact_legend=exact_legend,
                  model_legend=model_legend, narma_target_legend=target_legend,
                  section_headings=section_headings)
    return fig, panels


def render(output=OUTPUT):
    output = Path(output)
    emulation, task, dense = load_sources()
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        fig, _ = make_figure(emulation, task, dense)
        fig.savefig(output / f"{STEM}.pdf", metadata={"Title": "Memory emulation, prediction and exact limits",
            "Creator": "qrc_vs_qelm", "CreationDate": None, "ModDate": None})
        fig.savefig(output / f"{STEM}.png", dpi=300, metadata={"Software": "qrc_vs_qelm"})
        plt.close(fig)
    sources = [EMULATION, TASK, DENSE_REPLAY, Path(__file__), Path(colors.__file__), Path(roles.__file__),
               Path(__file__).with_name("narma_dense_replay.py"),
               Path(__file__).with_name("verify_narma_dense_replay.py"),
               Path(__file__).with_name("render_figure3_candidate.py"),
               Path(__file__).with_name("render_narma_additional.py"),
               Path(__file__).with_name("render_task_memory.py")]
    ledger = {
        "schema": "three_panel_memory_prediction_v1", "figure_size_inches": list(SIZE),
        "layout": {"column_left_inches": COL_LEFT, "plot_width_inches": PLOT_WIDTH,
                   "plot_height_inches": NARMA_TRACE_HEIGHT,
                   "upper_group_top_bottom_inches": [UPPER_TOP, UPPER_BOTTOM],
                   "lower_row_top_bottom_inches": [LOWER_TOP, LOWER_BOTTOM],
                   "panel_title_y_inches": PANEL_TITLE_Y, "section_title_y_inches": SECTION_TITLE_Y,
                   "legend_placement": "White framed boxes attached to the top-right edge of each lower plot; boxes extend above the axes to keep curves clear.",
                   "label_placement": "NARMA model names and target legend above data axes; laser condition titles outside axes; ten-input guide label below the history curve."},
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "parents": {"emulation": str(EMULATION.relative_to(ROOT)), "task": str(TASK.relative_to(ROOT)),
                    "dense_replay": str(DENSE_REPLAY.relative_to(ROOT))},
        "panels": {"a": "original 128-row laser-driven emulation; not forecasting",
                   "b": "12-root paired QRC gap above separate analytic convergence",
                   "c": "synthetic NARMA10 predictions stacked above 24-root history restriction"},
        "semantic_colors": {"external_target_and_input": roles.TARGET,
                            "narma_target_alpha": NARMA_TARGET_ALPHA,
                            "qrc": roles.QRC, "qelm": roles.QELM,
                            "full_history_reference": FULL_HISTORY_COLOR,
                            "memory_fast_loss": MEMORY_COLORS[4.],
                            "memory_slow_loss": MEMORY_COLORS[.2],
                            "different_earlier_inputs": EARLIER_INPUT_COLOR,
                            "shared_input_continuation": roles.INPUT},
        "replay_display": "Labels describe accessible history: QRC last d inputs, QRC full history, QELM last 4 inputs. Recent inputs include the current input. The restricted QRC is reset for every prediction; only the last d inputs are processed anew, with a separately refitted readout for each d. One computed point for every integer d=1..48; the linear display focuses on d=1..20, retaining all 24 original fresh roots at every d. Straight connections, no smoothing or interpolated observations. Full-history QRC median is a red dashed empirical reference, not a floor. QELM only has a four-input estimate here.",
        "narma_history_display": {"first_window": 1, "last_window": 20,
                                  "y_limits": [.40, .90], "guide_window": 10,
                                  "guide_interpretation": "At d=10 the quantized input at t-9 first enters the window for predicting y_(t+1). This exposes the lag used by NARMA10's explicit input interaction; it is not a claim of complete history or an isolated causal effect."},
        "omitted_from_display": "Quadratic curves omitted at author request; all 4/12-input results retained below and in manuscript prose.",
        "emulation": emulation, "narma_trace": task["traces"]["narma"],
        "narma_trace_display": {"start": 0, "stop_exclusive": NARMA_DISPLAY_STEPS,
                                "test_indices": task["traces"]["narma"]["test_indices"][:NARMA_DISPLAY_STEPS],
                                "description": "First 48 samples of the original fixed 96-sample excerpt; full traces retained, identical viewport and original y limits for both models."},
        "narma_all_rows": [r for r in task["all_rows"] if r["study"] == "narma"],
        "narma_aggregates": [r for r in task["aggregates"] if r["study"] == "narma"],
        "narma_candidate_rows": task["candidate_rows"],
        "narma_dense_replay": dense,
        "independent_verification": task["independent_verification"],
        "scope": "Full source excerpts retained; NARMA display zooms into the first 48 samples. No smoothing/cohort pooling. Root bands are 10th-90th percentiles, not CI. Replay costs extra evolution. Analytic floor does not bound sampled laser or NARMA errors.",
        "artifacts": {f"{STEM}.{ext}": digest(output / f"{STEM}.{ext}") for ext in ("pdf", "png")},
    }
    (output / f"{STEM}_source.json").write_text(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    render(parser.parse_args().output)
