"""Source-backed Figure 3 layout, available as preview or publication asset.

Combine the predeclared laser excerpt, paired-history memory control, and
existing analytic construction. All 360 N5 window-study observations remain
in the ledger and their original figures, not in the new memory panel.
No simulation, fitting, smoothing, cohort pooling, or excerpt selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter
import numpy as np
import pandas as pd

from laser_demonstration import Config, ROOT, digest, dump, load_laser, input_blocks, pair_histories
import paper_palette as colors
from render_laser_demonstration import save, validate_rows
from render_long_context_n5 import DEFAULT_ANALYSIS, DEFAULT_RESULTS, validated_source
from render_long_context_sharpness import COPIES
from sharpness_benchmarks import convergence_rows

SIZE = (6.221, 3.65)
STEM = "figure3_combined_candidate"
# Author-requested emphasis for the fixed missing-history floor only.
HISTORY_FLOOR_RED = "#E53935"
SERIES = (
    ("history", "Old history", colors.INK, "o"),
    ("qelm", "QELM", colors.CORAL, "s"),
    ("quadratic", "Quadratic delay", colors.VIOLET, "D"),
)
CAPTION = (
    "From measured inputs to a memory test. a A trained QELM follows the simulated "
    "QRC more closely with fast forgetting. This is emulation, not laser forecasting; "
    "percentages use the full test and training-output SD; output scales differ. "
    "b Different measured pasts are followed by the same measured continuation. "
    "The QRC output gap reveals the remaining influence of those pasts. From input "
    "four, the QELM windows agree. Curves and bands are medians and 10th–90th "
    "percentiles across 12 roots per rate, not confidence intervals. These deliberately "
    "spliced histories witness a pair-specific gap, not the global optimum; slow loss "
    "is not permanent memory. c More encoding copies approach an exactly known "
    "limit in a separate reference example: the readout is constructed, not learned, "
    "and the error is absolute, not NRMSE. The red dashed line marks the "
    "unavoidable error from missing history at this fixed window length."
)


def chart_style() -> dict:
    return {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
        "font.size": 8, "mathtext.fontset": "cm", "axes.labelsize": 8.5,
        "axes.labelweight": "normal", "axes.linewidth": .65,
        "text.color": colors.INK, "axes.labelcolor": colors.INK,
        "axes.edgecolor": colors.INK, "xtick.color": colors.INK,
        "ytick.color": colors.INK, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "legend.framealpha": 1,
        "legend.edgecolor": colors.INK, "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }


def add_axis(fig, left, bottom, width, height):
    ax = fig.add_axes([left / SIZE[0], bottom / SIZE[1], width / SIZE[0], height / SIZE[1]])
    ax.set_axisbelow(True)
    ax.grid(axis="both", color=colors.GRID, lw=.45, ls=(0, (4, 4)))
    ax.tick_params(length=2.5, width=.65, pad=2)
    return ax


def heading(fig, x, y, letter, title):
    fig.text(x / SIZE[0], y / SIZE[1], letter, fontsize=9, weight="bold", va="baseline")
    fig.text((x + .16) / SIZE[0], y / SIZE[1], title, fontsize=8.7, va="baseline")


def _legacy_layout(raw_excerpt, trajectories, errors, window_source, construction):
    """Pure layout function, exposed for binding and publication-size tests."""
    if len(raw_excerpt) != 128:
        raise ValueError("The predeclared excerpt contains 128 samples")
    counts = window_source.groupby(["teacher_gamma", "series", "window"]).size()
    expected = {(g, s, d) for g in (1., 4.) for s, *_ in SERIES for d in (2, 4, 6, 8, 10)}
    if len(window_source) != 360 or set(counts.index) != expected or not counts.eq(12).all():
        raise ValueError("Keep all 360 N5 root-series observations")
    if tuple(row["r"] for row in construction) != COPIES:
        raise ValueError("Keep every construction copy count")
    fig = plt.figure(figsize=SIZE)
    laser = [add_axis(fig, .55, bottom, 2.18, height)
             for bottom, height in ((2.87, .53), (1.73, 1.02), (.39, 1.06))]
    window = add_axis(fig, 3.42, 1.95, 2.68, 1.45)
    exact = add_axis(fig, 3.42, .39, 2.68, .94)
    heading(fig, .55, 3.51, "a", "Measured time series")
    heading(fig, 3.42, 3.51, "b", "More history")
    heading(fig, 3.42, 1.44, "c", "Approaching the exact limit")

    x = np.arange(128)
    laser[0].plot(x, raw_excerpt, color=colors.COBALT, lw=.8)
    laser[0].set_ylim(-12, 290)
    laser[0].set_yticks([0, 128, 256])
    laser[0].set_ylabel("Laser input", labelpad=4)
    for ax, rate, title in zip(laser[1:], Config().rates,
                               ("Fast memory loss", "Slow memory loss"), strict=True):
        y, q = trajectories[rate]
        if len(y) != 128 or len(q) != 128:
            raise ValueError("Do not drop or smooth trajectory values")
        ax.plot(x, y, color=colors.INK, lw=.85, marker=".", ms=1.7, label="QRC", zorder=3)
        ax.plot(x, q, color=colors.CORAL, lw=.65, marker="o", ms=2.4,
                mfc="white", mew=.65, markevery=3, label="QELM", zorder=4)
        lo, hi = min(np.min(y), np.min(q)), max(np.max(y), np.max(q))
        span = max(float(hi - lo), 1e-4)
        ax.set_ylim(lo - .18 * span, hi + .39 * span)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_ylabel("Output", labelpad=4)
        ax.text(.035, .95, title, ha="left", va="top", transform=ax.transAxes, fontsize=8)
        ax.text(.975, .045, f"full test: {100 * errors[rate]:.1f}%", ha="right",
                va="bottom", fontsize=7.3, color=colors.MUTED, transform=ax.transAxes)
    for ax in laser:
        ax.set_xlim(-1.5, 128.5)
        ax.set_xticks([0, 32, 64, 96, 127])
    for ax in laser[:-1]:
        ax.tick_params(labelbottom=False)
    laser[-1].set_xlabel("Time step in test excerpt", labelpad=4)
    handles, labels = laser[1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="center", bbox_to_anchor=(1.64 / SIZE[0], 1.59 / SIZE[1]),
                        ncol=2, fontsize=8.2, handlelength=1.5, handletextpad=.5,
                        borderpad=.42, columnspacing=1.2, fancybox=True)
    legend.get_frame().set_linewidth(.65)

    aggregate = window_source.groupby(["teacher_gamma", "series", "window"])["nrmse"].agg(
        median="median", lower=lambda s: s.quantile(.1), upper=lambda s: s.quantile(.9))
    for rate in (1., 4.):
        for name, label, color, marker in SERIES:
            rows = aggregate.loc[(rate, name)]
            median = rows["median"].to_numpy(float)
            line = window.errorbar(rows.index.to_numpy(float), median,
                yerr=np.vstack((median - rows["lower"], rows["upper"] - median)),
                color=color, marker=marker, linestyle="-" if rate == 1 else "--",
                mfc=color if rate == 1 else "white", ms=3.3, mew=.65,
                lw=.85, elinewidth=.6, capsize=1.3, capthick=.6)
            line.lines[0].set_gid(f"n5_{name}_gamma{rate:g}")
    window.set_yscale("log")
    window.set_ylim(1e-5, 2.5)
    window.set_xlim(1.65, 10.35)
    window.set_xticks([2, 4, 6, 8, 10])
    window.yaxis.set_major_locator(FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]))
    window.yaxis.set_minor_locator(FixedLocator([]))
    window.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}%"))
    window.yaxis.set_minor_formatter(NullFormatter())
    window.grid(which="minor", axis="y", color=colors.GRID, lw=.35, ls=(0, (4, 4)))
    window.axhline(.1, color=colors.MUTED, lw=.65, ls=(0, (4, 3)), zorder=1)
    window.set_xlabel(r"Window length $d$", labelpad=3)
    window.set_ylabel("Sampled error", labelpad=4)
    model_handles = [Line2D([], [], color=color, marker=marker, lw=.8, ms=3,
                            label=label) for _, label, color, marker in SERIES]
    model_legend = window.legend(handles=model_handles, loc="lower left", fontsize=7.2,
                                  handlelength=1.5, handletextpad=.5, borderpad=.35,
                                  labelspacing=.25, fancybox=True)
    model_legend.get_frame().set_linewidth(.6)
    window.add_artist(model_legend)
    rate_handles = [Line2D([], [], color=colors.INK, lw=.8, marker="o", ms=3,
                          mfc=colors.INK if rate == 1 else "white",
                          ls="-" if rate == 1 else "--", label=rf"$\gamma_T={rate}$")
                    for rate in (1, 4)]
    rate_legend = window.legend(handles=rate_handles, loc="upper right", ncol=2,
                                 fontsize=7.6, handlelength=1.8, borderpad=.3,
                                 handletextpad=.4, columnspacing=.75, frameon=False)

    exact.set_xscale("log", base=2)
    exact.set_xlim(.8, 320)
    exact.set_ylim(.064, .315)
    exact.set_xticks([1, 4, 16, 64, 256], labels=[1, 4, 16, 64, 256])
    exact.set_yticks([.1, .2, .3])
    exact.plot([row["r"] for row in construction], [row["error"] for row in construction],
               color=colors.COBALT, marker="o", ms=3.1, lw=1, label="Constructed QELM")
    floor = exact.axhline(construction[0]["floor"], color=HISTORY_FLOOR_RED, lw=1.1,
                         ls=(0, (4, 3)), label="Unavoidable window error", zorder=1.9)
    floor.set_gid("missing_history_floor")
    exact.set_ylabel("Worst-case\nabsolute error", labelpad=4)
    exact.set_xlabel(r"Encoding copies per input $r$", labelpad=4)
    exact_legend = exact.legend(loc="upper right", fontsize=7.4, handlelength=1.8,
                                borderpad=.35, labelspacing=.3, fancybox=True)
    exact_legend.get_frame().set_linewidth(.6)
    return fig, {"laser": laser, "window": window, "exact": exact,
                 "model_legend": model_legend, "rate_legend": rate_legend}


def memory_statistics(paired):
    """Keep all seeds and steps; the displayed first 48 steps match the old control view."""
    if set(paired) != set(Config().rates):
        raise ValueError("Both predeclared loss rates are required")
    result = {}
    for rate, values in paired.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (12, 128, 4) or not np.isfinite(values).all():
            raise ValueError("Keep all 12 seeds, 128 shared inputs and four output traces")
        ya, yb, qa, qb = np.moveaxis(values, -1, 0)
        if not np.allclose(qa[:, 3:], qb[:, 3:], rtol=0, atol=1e-10):
            raise ValueError("The four-input predictions must agree")
        gap = abs(ya - yb)
        worst = np.maximum(abs(ya - qa), abs(yb - qb))
        if np.any(worst[:, 3:] + 1e-10 < gap[:, 3:] / 2):
            raise ValueError("Paired scalar lower-bound check failed")
        result[rate] = {"gap": gap, "median": np.median(gap, axis=0),
                        "low": np.quantile(gap, .1, axis=0),
                        "high": np.quantile(gap, .9, axis=0)}
    return result


def build_figure(raw_excerpt, trajectories, errors, window_source, construction,
                 paired, history_inputs):
    fig, panels = _legacy_layout(raw_excerpt, trajectories, errors, window_source, construction)
    panels.pop("window").remove()
    panels.pop("model_legend")
    panels.pop("rate_legend")
    for text in fig.texts:
        if text.get_text() == "More history":
            text.set_text("Memory of earlier inputs")
    # The in-panel regime labels echo the shared model legend without hiding traces.
    for ax in panels["laser"][1:]:
        ax.texts[0].set_bbox(dict(boxstyle="round,pad=.25", fc="white", ec=colors.INK, lw=.6))
    memory = add_axis(fig, 3.42, 1.94, 2.68, 1.17)
    stats = memory_statistics(paired)
    x = np.arange(1, 49)
    for rate, label, color, marker, ls in (
            (4., "Fast loss", colors.COBALT, "o", "-"),
            (.2, "Slow loss", colors.VIOLET, "D", "--")):
        s = stats[rate]
        memory.fill_between(x, s["low"][:48], s["high"][:48], color=color, alpha=.15, lw=0)
        line, = memory.plot(x, s["median"][:48], color=color, lw=.95, ls=ls,
                            marker=marker, ms=2.3, markevery=4, label=label)
        line.set_gid(f"memory_gap_gamma{rate:g}")
    memory.set_xlim(.2, 49)
    high = max(s["high"][:48].max() for s in stats.values())
    memory.set_ylim(-.015 * high, 1.18 * high)
    memory.set_xticks([1, 4, 16, 32, 48])
    memory.yaxis.set_major_locator(MaxNLocator(4))
    memory.set_ylabel("QRC output gap", labelpad=4)
    memory.set_xlabel("Shared inputs received", labelpad=3)
    memory.axvline(4, color=colors.MUTED, ls=(0, (3, 3)), lw=.7)
    memory.text(.985, .43, "Same window from input 4", transform=memory.transAxes,
                ha="right", va="bottom", color=colors.MUTED, fontsize=7.2)
    legend = memory.legend(loc="upper right", fontsize=7.5, handlelength=1.7,
                           labelspacing=.25, borderpad=.35, fancybox=True)
    legend.get_frame().set_linewidth(.6)

    # Two actual quantized laser histories, offset only to make the rows distinct.
    # The framed mini-plot straddles the upper edge like an explanatory legend.
    histories = np.asarray(history_inputs, dtype=float)
    if histories.shape != (2, 256) or not np.array_equal(histories[0, 128:], histories[1, 128:]):
        raise ValueError("History inset must show two 128-input prefixes and their shared 128-input continuation")
    inset = add_axis(fig, 3.54, 3.035, 2.56, .40)
    inset.set_zorder(8)
    inset.set_facecolor("white")
    inset.grid(False)
    for i, row in enumerate(histories):
        baseline = .18 + (1 - i) * .48
        inset.plot(np.arange(128), baseline + .31 * row[:128], color=colors.INK,
                   lw=.45, ls="-" if i == 0 else "--")
        inset.plot(np.arange(128, 256), baseline + .31 * row[128:],
                   color=colors.COBALT, lw=.5)
    inset.axvline(127.5, color=colors.MUTED, ls=(0, (3, 2)), lw=.5)
    inset.text(.25, .97, "Different earlier inputs", transform=inset.transAxes,
               fontsize=7.3, ha="center", va="top")
    inset.text(.75, .97, "Same inputs from here on", transform=inset.transAxes,
               fontsize=7.3, ha="center", va="top")
    inset.set_xlim(-6, 261)
    inset.set_ylim(.1, 1.43)
    inset.set_xticks([])
    inset.set_yticks([])
    panels.update(memory=memory, history_inset=inset, memory_legend=legend)
    return fig, panels


def render(source: Path, output: Path, data: Path, results=DEFAULT_RESULTS, analysis=DEFAULT_ANALYSIS,
           *, publication=False):
    cfg = Config()
    summary = json.loads((source / "summary.json").read_text())
    validate_rows(source, summary)
    specification = json.loads((source / "run_specification.json").read_text())
    for filename, key in (("laser_demonstration.py", "runner_sha256"),):
        if digest(Path(__file__).with_name(filename)) != specification[key]:
            raise ValueError("The predeclared numerical runner changed")
    if digest(ROOT / "pra_extension/LASER_DEMONSTRATION_PROTOCOL.md") != specification["protocol_sha256"]:
        raise ValueError("The predeclared laser protocol changed")
    raw = load_laser(data)
    excerpt = slice(cfg.washout, cfg.washout + 128)
    trajectories, errors = {}, {}
    for rate in cfg.rates:
        with np.load(source / f"seed{cfg.seeds[0]}_gamma{rate:g}.npz") as rows:
            trajectories[rate] = (rows["natural_target"][excerpt].copy(),
                                  rows["natural_prediction"][excerpt].copy())
        errors[rate] = next(r["test_nrmse_training_sd"] for r in summary["results"]
                            if r["seed"] == cfg.seeds[0] and r["gamma"] == rate)
    offset = sum(cfg.block_lengths[:2]) + cfg.washout
    raw_excerpt = raw[offset:offset + 128]
    blocks, _ = input_blocks(raw, cfg)
    histories = np.array(pair_histories(blocks[2], cfg))
    paired = {}
    for rate in cfg.rates:
        rows_by_seed = []
        for seed in cfg.seeds:
            with np.load(source / f"seed{seed}_gamma{rate:g}.npz") as rows:
                rows_by_seed.append(np.column_stack([rows[key] for key in
                    ("pair_y_a", "pair_y_b", "pair_q_a", "pair_q_b")]))
        paired[rate] = np.array(rows_by_seed)
    memory = memory_statistics(paired)
    windows = validated_source(results, analysis)
    construction = convergence_rows(d=8, eta=.8, beta=2., copies=COPIES)
    stem = "fig04_emulation_evidence" if publication else STEM
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(chart_style()):
        fig, _ = build_figure(raw_excerpt, trajectories, errors, windows, construction, paired, histories)
        artifacts = save(fig, output, stem)
    caption_path = output / f"{stem}_caption.txt"
    caption_path.write_text(CAPTION + "\n")
    artifacts[caption_path.name] = digest(caption_path)
    manifest = {
        "figure_size_inches": list(SIZE),
        "status": "author_selected_publication_asset" if publication else "candidate_only",
        "paper_modified": False, "laser_summary_sha256": digest(source / "summary.json"),
        "laser_run_specification": specification,
        "laser_all_seed_summary": {key: value for key, value in summary.items() if key != "files_sha256"},
        "laser_preprocessing": json.loads((source / "preprocessing.json").read_text()),
        "laser_data_sha256": digest(data), "laser_seed": cfg.seeds[0],
        "laser_test_excerpt": [cfg.washout, cfg.washout + 128],
        "laser_input": raw_excerpt.tolist(),
        "laser_outputs": {str(rate): {"qrc": ys[0].tolist(), "qelm": ys[1].tolist(),
                                        "full_test_nrmse_training_sd": errors[rate]}
                          for rate, ys in trajectories.items()},
        "window_rows": windows.to_dict(orient="records"),
        "paired_outputs": {str(rate): values.tolist() for rate, values in paired.items()},
        "paired_seed_order": list(cfg.seeds), "paired_display_steps": [1, 48],
        "paired_history_inputs": histories.tolist(),
        "memory_statistics": {str(rate): {key: value.tolist() for key, value in s.items()}
                              for rate, s in memory.items()},
        "convergence_rows": construction,
        "retained": "Fixed laser excerpt; all 24 paired controls; all analytic copy counts; all N5 rows in ledger",
        "omitted_from_this_layout": "N5 window and feature panels; original source, figures and text results retained",
        "metrics": {"a": "observable expectations; annotation=test RMSE / training-output SD",
                    "b": "absolute paired QRC output gap; 10th-90th seed percentiles, not CI or global D_d",
                    "c": "absolute worst-case scalar error; specified readout, not trained"},
        "style_reference": "User-supplied arXiv:2608.25511v1, Fig.5, p.7; grouping only",
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), Path(__file__).with_name("render_laser_demonstration.py"),
            Path(__file__).with_name("render_long_context_n5.py"),
            Path(__file__).with_name("render_long_context_sharpness.py"),
            Path(__file__).with_name("sharpness_benchmarks.py"),
            Path(__file__).with_name("paper_palette.py"))},
        "artifacts": artifacts,
    }
    dump(output / f"{stem}_source.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results_laser_demonstration_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results_laser_demonstration_v1/figure3_candidate")
    parser.add_argument("--data", type=Path, default=ROOT / "data/santa_fe_laser/SF_Acont.dat")
    parser.add_argument("--publication", action="store_true",
                        help="Create the author-selected asset; use --output paper/figures/emulation_evidence")
    args = parser.parse_args()
    manifest = render(args.source, args.output, args.data, publication=args.publication)
    print(json.dumps({"output": str(args.output), "artifacts": manifest["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
