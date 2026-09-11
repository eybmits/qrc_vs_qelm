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
from render_figure3_candidate import build_figure, chart_style, SIZE
from render_narma_additional import load_source as load_task_source, digest
from render_task_memory import _summary

ROOT = Path(__file__).resolve().parents[2]
EMULATION = ROOT / "paper/figures/emulation_evidence/fig04_emulation_evidence_source.json"
TASK = ROOT / "paper/figures/task_memory/fig04_task_memory_source.json"
OUTPUT = ROOT / "paper/figures/memory_prediction"
STEM = "fig04_memory_prediction"


def load_sources():
    emulation = json.loads(EMULATION.read_text())
    for relative, expected in emulation["sources"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Changed emulation dependency: {relative}")
    for name, expected in emulation["artifacts"].items():
        if digest(EMULATION.parent / name) != expected:
            raise ValueError(f"Changed emulation artifact: {name}")
    task = load_task_source(TASK)
    return emulation, task


def _position(ax, left, bottom, width, height):
    ax.set_position([left / SIZE[0], bottom / SIZE[1], width / SIZE[0], height / SIZE[1]])


def _axis(fig, left, bottom, width, height, name):
    ax = fig.add_axes([left / SIZE[0], bottom / SIZE[1], width / SIZE[0], height / SIZE[1]], label=name)
    ax.set_axisbelow(True)
    ax.grid(color=colors.GRID, lw=.4, ls=(0, (4, 4)))
    ax.tick_params(length=2.3, width=.6, pad=2)
    return ax


def make_figure(emulation, task):
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
    for text in list(fig.texts):
        text.remove()
    for legend in list(fig.legends):
        legend.remove()
    for x, letter, title in ((.44, "a", "Laser-driven emulation"),
                             (2.53, "b", "NARMA10 prediction"),
                             (4.59, "c", "Memory and exact limit")):
        fig.text(x / SIZE[0], 3.52 / SIZE[1], letter, fontsize=9, weight="bold")
        fig.text((x + .15) / SIZE[0], 3.52 / SIZE[1], title, fontsize=8)

    # a: Keep every one of the original 128 excerpt samples.
    for ax, bottom, height in zip(panels["laser"], (2.84, 1.76, .39), (.5, .94, 1.06), strict=True):
        _position(ax, .47, bottom, 1.47, height)
        ax.set_xticks([0, 64, 127])
        ax.set_ylabel(ax.get_ylabel(), labelpad=3)
    panels["laser"][-1].set_xlabel("Test step (excerpt)", labelpad=3)
    for ax in panels["laser"][1:]:
        ax.texts[0].set_fontsize(7.5)
        ax.texts[0].set_text(ax.texts[0].get_text().replace(" memory", ""))
        ax.texts[1].set_fontsize(7.5)
        ax.texts[1].set_text(ax.texts[1].get_text().replace("full test:", "test:"))
    handles, labels = panels["laser"][1].get_legend_handles_labels()
    model_legend = fig.legend(handles, labels, loc="center", ncol=2,
        bbox_to_anchor=(1.19 / SIZE[0], 1.61 / SIZE[1]), fontsize=7.5,
        handlelength=1.2, handletextpad=.4, columnspacing=.8, borderpad=.35)
    model_legend.get_frame().set_linewidth(.6)

    # c: Preserve the measured-input memory control AND its separate analytic reference.
    _position(panels["memory"], 4.59, 1.94, 1.51, 1.06)
    panels["memory"].set_xticks([1, 16, 32, 48])
    panels["memory"].set_xlabel("Shared inputs received", labelpad=3)
    panels["memory"].set_ylabel("QRC output gap", labelpad=3)
    panels["memory"].texts[0].set_text("Same window\nfrom input 4")
    panels["memory"].texts[0].set_fontsize(7.5)
    panels["memory"].texts[0].set_y(.31)
    _position(panels["history_inset"], 4.59, 2.98, 1.51, .46)
    panels["history_inset"].set_ylim(.1, 1.9)
    for text, words in zip(panels["history_inset"].texts,
        ("Different earlier\ninputs", "Same inputs\nfrom here on"), strict=True):
        text.set_text(words)
        text.set_fontsize(7.5)
    _position(panels["exact"], 4.59, .39, 1.51, .94)
    panels["exact"].set_xticks([1, 16, 256], labels=[1, 16, 256])
    panels["exact"].set_xlabel("Copies per input", labelpad=3)
    panels["exact"].set_ylabel("Worst-case\nabsolute error", labelpad=3)
    panels["exact"].get_legend().remove()
    exact_handles, _ = panels["exact"].get_legend_handles_labels()
    exact_legend = panels["exact"].legend(exact_handles,
        ["Constructed QELM", "Unavoidable\nwindow error"], loc="upper right",
        fontsize=7.5, handlelength=1.2, handletextpad=.35, borderpad=.3,
        labelspacing=.25)
    exact_legend.get_frame().set_linewidth(.6)
    fig.text(4.59 / SIZE[0], 1.47 / SIZE[1], "Analytical benchmark", fontsize=8)

    # b: The fixed fresh NARMA excerpt; separate raw targets are never averaged.
    trace = task["traces"]["narma"]
    values = np.asarray([trace[k] for k in ("target", "qrc", "qelm")])
    lo, hi = values.min(), values.max()
    narma = {}
    for model, bottom in (("qrc", 2.73), ("qelm", 1.96)):
        ax = _axis(fig, 2.53, bottom, 1.48, .61, "narma_" + model)
        narma[model] = ax
        ax.plot(np.arange(96), trace["target"], color=colors.COBALT, lw=.75, label="Target")
        ax.plot(np.arange(96), trace[model], color=colors.INK if model == "qrc" else colors.CORAL,
            lw=.7, ls="--" if model == "qrc" else "-", marker=None if model == "qrc" else "o",
            ms=2.0, mfc="white", mew=.5, markevery=8, label=model.upper())
        ax.set_ylim(lo - .08 * (hi-lo), hi + .30 * (hi-lo))
        ax.set_yticks([.18, .21, .24])
        ax.set_xlim(0, 95)
        ax.set_xticks([0, 48, 95])
        ax.set_ylabel("Output", labelpad=3)
        ax.text(.97, .96, model.upper(), ha="right", va="top", fontsize=7.5, transform=ax.transAxes)
        if model == "qrc":
            ax.tick_params(labelbottom=False)
            ax.legend(handles=[ax.lines[0]], loc="upper left", frameon=False,
                fontsize=7.5, handlelength=1.0, handletextpad=.3, borderpad=.15)
        else:
            ax.set_xlabel("Test step (excerpt)", labelpad=3)
    history = _axis(fig, 2.53, .39, 1.48, .75, "narma_history")
    narma["history"] = history
    replay = _summary(task, "narma", "selected", "replay")
    windows = np.array([r["window"] for r in replay])
    med, low, high = [np.array([r[k] for r in replay]) for k in ("median", "p10", "p90")]
    history.fill_between(windows, low, high, color=colors.VIOLET, alpha=.15, lw=0)
    history.plot(windows, med, color=colors.VIOLET, marker="o", ms=2.5, lw=.85, label="Replay")
    qrc = _summary(task, "narma", "selected", "qrc")[0]
    history.axhspan(qrc["p10"], qrc["p90"], color=colors.INK, alpha=.06, lw=0)
    history.axhline(qrc["median"], color=colors.INK, lw=.85, label="QRC (full)")
    qelm = _summary(task, "narma", "selected", "qelm")[0]
    history.errorbar([4], [qelm["median"]],
        yerr=[[qelm["median"]-qelm["p10"]], [qelm["p90"]-qelm["median"]]],
        color=colors.CORAL, marker="o", ms=3.5, lw=.8, capsize=1.5, label="QELM(4)", zorder=5)
    classical = _summary(task, "narma", "selected", "quadratic")
    med, low, high = [np.array([r[k] for r in classical]) for k in ("median", "p10", "p90")]
    history.errorbar([r["window"] for r in classical], med, yerr=[med-low, high-med],
        color=colors.MUTED, marker="D", mfc="white", ms=3, mew=.7, ls=":", lw=.8,
        capsize=1.5, label="Quadratic", zorder=4)
    history.set_xscale("log", base=2)
    history.set_xlim(.85, 56)
    history.set_xticks([1, 4, 12, 48], labels=[1, 4, 12, 48])
    history.set_ylim(0, .93)
    history.set_yticks([0, .4, .8])
    history.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:g}%"))
    history.set_xlabel("Replayed inputs", labelpad=3)
    history.set_ylabel("Test NRMSE", labelpad=3)
    narma_legend = fig.legend(handles=[
        Line2D([], [], color=colors.VIOLET, marker="o", ms=2.5, lw=.85, label="Replay"),
        Line2D([], [], color=colors.INK, lw=.85, label="QRC (full)"),
        Line2D([], [], color=colors.CORAL, marker="o", ms=3, lw=0, label="QELM(4)"),
        Line2D([], [], color=colors.MUTED, marker="D", mfc="white", ms=2.5, lw=.8, ls=":", label="Quadratic")],
        loc="center", bbox_to_anchor=(3.27 / SIZE[0], 1.38 / SIZE[1]), ncol=2,
        fontsize=7.5, handlelength=1.05, handletextpad=.35, columnspacing=.6,
        labelspacing=.3, borderpad=.3)
    narma_legend.get_frame().set_linewidth(.6)
    panels.update(narma=narma, narma_legend=narma_legend, exact_legend=exact_legend,
                  model_legend=model_legend)
    return fig, panels


def render(output=OUTPUT):
    output = Path(output)
    emulation, task = load_sources()
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        fig, _ = make_figure(emulation, task)
        fig.savefig(output / f"{STEM}.pdf", metadata={"Title": "Memory emulation, prediction and exact limits",
            "Creator": "qrc_vs_qelm", "CreationDate": None, "ModDate": None})
        fig.savefig(output / f"{STEM}.png", dpi=300, metadata={"Software": "qrc_vs_qelm"})
        plt.close(fig)
    sources = [EMULATION, TASK, Path(__file__), Path(colors.__file__),
               Path(__file__).with_name("render_figure3_candidate.py"),
               Path(__file__).with_name("render_narma_additional.py"),
               Path(__file__).with_name("render_task_memory.py")]
    ledger = {
        "schema": "three_panel_memory_prediction_v1", "figure_size_inches": list(SIZE),
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "parents": {"emulation": str(EMULATION.relative_to(ROOT)), "task": str(TASK.relative_to(ROOT))},
        "panels": {"a": "original 128-row laser-driven emulation; not forecasting",
                   "b": "synthetic NARMA10 predictions stacked above 24-root history restriction",
                   "c": "12-root paired QRC gap above separate analytic convergence"},
        "emulation": emulation, "narma_trace": task["traces"]["narma"],
        "narma_all_rows": [r for r in task["all_rows"] if r["study"] == "narma"],
        "narma_aggregates": [r for r in task["aggregates"] if r["study"] == "narma"],
        "narma_candidate_rows": task["candidate_rows"],
        "independent_verification": task["independent_verification"],
        "scope": "Unchanged excerpts; no smoothing/cohort pooling. Root bands are 10th-90th percentiles, not CI. Replay costs extra evolution. Analytic floor does not bound sampled laser or NARMA errors.",
        "artifacts": {f"{STEM}.{ext}": digest(output / f"{STEM}.{ext}") for ext in ("pdf", "png")},
    }
    (output / f"{STEM}_source.json").write_text(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    render(parser.parse_args().output)
