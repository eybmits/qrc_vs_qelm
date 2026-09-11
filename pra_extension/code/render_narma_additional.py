#!/usr/bin/env python3
"""Optional NARMA-only view; no simulation, selection, or manuscript mutation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

import paper_palette as colors
from render_task_memory import _style, _summary, validate_payload

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "paper/figures/task_memory/fig04_task_memory_source.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/narma_additional"
FIGURE_SIZE = (6.221, 2.4)
STEM = "narma_additional"
CAPTION = (
    "Optional NARMA10 supplement, not part of the active manuscript. Left: actual "
    "synthetic target and fitted QRC/QELM predictions, first fresh root 921000, first "
    "96 scored test rows; no smoothing or averaging of different targets. Right: "
    "test NRMSE medians and 10th-90th percentiles across all 24 fresh roots, not "
    "confidence intervals. Reset/replay restricts the same QRC and refits its readout; "
    "longer replay costs more evolution steps. Four-input QELM and four-/twelve-input "
    "quadratic controls remain visible. Inputs drive the NARMA recurrence; neither "
    "model receives past targets. Exact expectations; no quantum-advantage or "
    "worst-case-bound claim."
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(source=DEFAULT_SOURCE):
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    validate_payload(payload)
    if not payload["independent_verification"]["verification_passed"]:
        raise ValueError("Expected independently verified fresh replication")
    for relative, expected in payload["sources"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Changed parent source dependency: {relative}")
    return payload


def _axis(fig, bounds, name):
    left, bottom, width, height = bounds
    ax = fig.add_axes([left / FIGURE_SIZE[0], bottom / FIGURE_SIZE[1],
                       width / FIGURE_SIZE[0], height / FIGURE_SIZE[1]], label=name)
    ax.set_axisbelow(True)
    ax.grid(axis="both", color=colors.GRID, linewidth=.4, linestyle=(0, (4, 4)))
    ax.tick_params(pad=2)
    return ax


def make_figure(payload):
    validate_payload(payload)
    _style()
    fig = plt.figure(figsize=FIGURE_SIZE)
    trace = payload["traces"]["narma"]
    values = np.asarray([trace[k] for k in ("target", "qrc", "qelm")])
    lo, hi = float(values.min()), float(values.max())
    padding = .09 * (hi - lo)
    axes = {}
    for model, bottom in (("qrc", 1.23), ("qelm", .39)):
        ax = _axis(fig, (.49, bottom, 2.95, .56), model)
        axes[model] = ax
        ax.plot(np.arange(96), trace["target"], color=colors.COBALT, linewidth=.95, label="target")
        ax.plot(np.arange(96), trace[model], color=colors.INK if model == "qrc" else colors.CORAL,
                linewidth=.8, linestyle=(0, (3, 2)) if model == "qrc" else "-",
                marker=None if model == "qrc" else "o", markerfacecolor="white",
                markersize=2.3, markeredgewidth=.5, markevery=8, label=model.upper())
        ax.set_ylim(lo - padding, hi + padding)
        ax.set_yticks([.18, .21, .24])
        ax.set_xlim(0, 95)
        ax.set_xticks([0, 24, 48, 72, 95])
        ax.set_ylabel("output", labelpad=3)
        ax.set_title(model.upper(), loc="right", pad=2, fontsize=7.5)
        if model == "qrc":
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("test step (excerpt)", labelpad=3)
    ax = _axis(fig, (4.01, .39, 2.12, 1.40), "history")
    axes["history"] = ax
    replay = _summary(payload, "narma", "selected", "replay")
    windows = [r["window"] for r in replay]
    median, lower, upper = (np.asarray([r[key] for r in replay]) for key in ("median", "p10", "p90"))
    ax.fill_between(windows, lower, upper, color=colors.VIOLET, alpha=.12, linewidth=0)
    ax.plot(windows, median, color=colors.VIOLET, marker="o", markersize=3.5, linewidth=1,
            markeredgewidth=.6, label="replay/refit")
    qrc = _summary(payload, "narma", "selected", "qrc")[0]
    ax.axhspan(qrc["p10"], qrc["p90"], color=colors.INK, alpha=.06, linewidth=0)
    ax.axhline(qrc["median"], color=colors.INK, linewidth=.9, label="QRC (full)")
    qelm = _summary(payload, "narma", "selected", "qelm")[0]
    ax.errorbar([4], [qelm["median"]], yerr=[[qelm["median"]-qelm["p10"]], [qelm["p90"]-qelm["median"]]],
                color=colors.CORAL, marker="o", markersize=4.5, markeredgewidth=.7,
                linestyle="none", capsize=2, elinewidth=.8, label="QELM(4)", zorder=5)
    classical = _summary(payload, "narma", "selected", "quadratic")
    median, lower, upper = (np.asarray([r[key] for r in classical]) for key in ("median", "p10", "p90"))
    ax.errorbar([r["window"] for r in classical], median, yerr=[median-lower, upper-median],
                color=colors.MUTED, marker="D", markerfacecolor="white", markersize=4,
                markeredgewidth=.8, linestyle=(0, (1, 2)), linewidth=.85, capsize=2,
                elinewidth=.8, label="quadratic", zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xlim(.85, 56)
    ax.set_xticks([1, 4, 8, 12, 24, 48], labels=["1", "4", "8", "12", "24", "48"])
    ax.set_ylim(0, .93)
    ax.set_yticks([0, .4, .8])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100*value:g}%"))
    ax.set_ylabel("test NRMSE", labelpad=3)
    ax.set_xlabel("replayed inputs $d$", labelpad=3)
    fig.text(.49 / FIGURE_SIZE[0], 2.26 / FIGURE_SIZE[1], "a", fontsize=10, fontweight="bold", va="center")
    fig.text(.69 / FIGURE_SIZE[0], 2.26 / FIGURE_SIZE[1], "NARMA10 (synthetic)", fontsize=9, va="center")
    fig.text(4.01 / FIGURE_SIZE[0], 2.26 / FIGURE_SIZE[1], "b", fontsize=10, fontweight="bold", va="center")
    fig.text(4.21 / FIGURE_SIZE[0], 2.26 / FIGURE_SIZE[1], "Accessible history", fontsize=9, va="center")
    left = fig.legend(handles=[Line2D([], [], color=colors.COBALT, lw=.95, label="target"),
        Line2D([], [], color=colors.INK, lw=.8, linestyle="--", label="QRC"),
        Line2D([], [], color=colors.CORAL, lw=.8, marker="o", markerfacecolor="white", markersize=3, label="QELM")],
        loc="center", bbox_to_anchor=(1.90/FIGURE_SIZE[0], 2.04/FIGURE_SIZE[1]), ncol=3,
        frameon=False, handlelength=1.6, handletextpad=.5, columnspacing=1.1)
    right = fig.legend(handles=[Line2D([], [], color=colors.VIOLET, marker="o", markersize=3, lw=1, label="replay/refit"),
        Line2D([], [], color=colors.INK, lw=.9, label="QRC (full)"),
        Line2D([], [], color=colors.CORAL, marker="o", markersize=3.5, lw=0, label="QELM(4)"),
        Line2D([], [], color=colors.MUTED, marker="D", markerfacecolor="white", markersize=3, lw=.8, linestyle=":", label="quadratic")],
        loc="center", bbox_to_anchor=(5.08/FIGURE_SIZE[0], 2.01/FIGURE_SIZE[1]), ncol=2,
        frameon=False, handlelength=1.3, handletextpad=.4, columnspacing=1.0, labelspacing=.3)
    return fig, axes, (left, right)


def render(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    source, output = Path(source), Path(output)
    payload = load_source(source)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        fig, _, _ = make_figure(payload)
        fig.savefig(output / f"{STEM}.pdf", metadata={"Title": "Optional NARMA10 additional evidence",
                    "Creator": "qrc_vs_qelm NARMA supplementary renderer", "CreationDate": None, "ModDate": None})
        fig.savefig(output / f"{STEM}.png", dpi=300, metadata={"Software": "qrc_vs_qelm"})
        plt.close(fig)
    paths = [Path(__file__), Path(__file__).with_name("render_task_memory.py"), Path(colors.__file__), source]
    sources = {str(p.resolve().relative_to(ROOT)): digest(p) for p in paths}
    ledger = {"schema_version": 1, "status": "optional_addition_not_active_manuscript",
        "sources": sources, "parent_ledger": str(source.resolve().relative_to(ROOT)),
        "figure_size_inches": list(FIGURE_SIZE), "caption": CAPTION,
        "trace": payload["traces"]["narma"],
        "root_rows": [r for r in payload["all_rows"] if r["study"] == "narma"],
        "candidate_rows": payload["candidate_rows"],
        "aggregates": [r for r in payload["aggregates"] if r["study"] == "narma"],
        "verification": payload["independent_verification"],
        "scope": {k: payload["scope"][k] for k in ("narma", "traces", "interval", "history", "claims")},
        "artifacts": {f"{STEM}.{ext}": digest(output / f"{STEM}.{ext}") for ext in ("pdf", "png")}}
    (output / "source.json").write_text(json.dumps(ledger, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.source, args.output)
