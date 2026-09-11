"""Static, source-backed Cobalt figure candidates for the laser demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import scipy

from laser_demonstration import Config, ROOT, digest, dump, load_laser
from paper_palette import COBALT, CORAL, GRID, INK, MUTED

PAPER_COLUMN_INCHES = (6.221 - 0.8 / 2.54) / 2
PAPER_FIGURE_HEIGHT = 3.55


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "legend.framealpha": 1, "legend.edgecolor": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })


def axis_style(ax, label: str) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="both", color=GRID, lw=0.7, ls=(0, (4, 4)))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.text(-0.09, 1.04, label, transform=ax.transAxes, fontsize=12, weight="bold")
    ax.margins(x=0.015, y=0.2)


def save(fig, output: Path, stem: str) -> dict:
    pdf, png = output / f"{stem}.pdf", output / f"{stem}.png"
    fig.savefig(pdf, metadata={"Title": stem.replace("_", " "),
                              "Creator": "qrc_vs_qelm reproducible renderer",
                              "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=220)
    plt.close(fig)
    return {p.name: digest(p) for p in (pdf, png)}


def validate_rows(source: Path, summary: dict) -> None:
    cfg = Config()
    cells = summary["results"]
    if len(cells) != len(cfg.seeds) * len(cfg.rates):
        raise ValueError("Incomplete coverage: do not plot partial seed selection")
    keys = {(cell["seed"], cell["gamma"]) for cell in cells}
    if keys != {(seed, rate) for seed in cfg.seeds for rate in cfg.rates}:
        raise ValueError("Unregistered seed/rate coverage")
    for filename, expected in summary["files_sha256"].items():
        if digest(source / filename) != expected:
            raise ValueError(f"Source artifact hash mismatch: {filename}")
    for cell in cells:
        with np.load(source / cell["array_file"]) as rows:
            residual = rows["natural_target"][cfg.washout:] - rows["natural_prediction"][cfg.washout:]
            recomputed = float(np.sqrt(np.mean(residual ** 2)))
            if not np.isclose(recomputed, cell["test_rmse"], rtol=1e-12, atol=1e-14):
                raise ValueError("Array-to-summary RMSE disagreement")
            j = cfg.window - 1
            ya, yb, qa, qb = [rows[key][j:] for key in ("pair_y_a", "pair_y_b", "pair_q_a", "pair_q_b")]
            if not np.allclose(qa, qb, rtol=0, atol=1e-10):
                raise ValueError("Same windows did not produce the same predictions")
            if np.any(np.maximum(abs(ya - qa), abs(yb - qb)) + 1e-10 < abs(ya - yb) / 2):
                raise ValueError("Pointwise paired lower bound violated")


def natural_plot(source: Path, output: Path, summary: dict, raw: np.ndarray) -> dict:
    cfg, seed = Config(), Config().seeds[0]
    segment = slice(cfg.washout, cfg.washout + 128)
    x = np.arange(128)
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.7), sharex=True,
                             gridspec_kw={"height_ratios": [0.75, 1.0, 1.0]})
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.16, top=0.86, hspace=0.38)
    fig.text(0.13, 0.958, "A measured time series, two memory regimes", fontsize=15)
    fig.text(0.13, 0.921, "Santa Fe laser  |  four qubits  |  four-input window", color=MUTED, fontsize=11)
    offset = sum(cfg.block_lengths[:2])
    axes[0].plot(x, raw[offset + cfg.washout:offset + cfg.washout + 128],
                 color=COBALT, lw=1.3)
    axes[0].set_ylabel("Laser intensity\n(recorded units)")
    axes[0].set_title("Measured input", loc="left", fontsize=11, pad=10)
    for ax, rate, title in zip(axes[1:], cfg.rates,
                               ("Fast memory loss", "Slow memory loss")):
        with np.load(source / f"seed{seed}_gamma{rate:g}.npz") as rows:
            ax.plot(x, rows["natural_target"][segment], color=INK, lw=1.3,
                    marker=".", markersize=3, label="QRC output", zorder=3)
            ax.plot(x, rows["natural_prediction"][segment], color=CORAL, lw=0.9,
                    marker="o", markersize=4.0, markerfacecolor="white", markevery=2,
                    label="Trained QELM", zorder=4)
        cell = next(r for r in summary["results"] if r["seed"] == seed and r["gamma"] == rate)
        ax.set_title(title, loc="left", fontsize=11, pad=10)
        ax.set_title(f"Full-test error: {100 * cell['test_nrmse_training_sd']:.1f}%", loc="right",
                     fontsize=10, color=MUTED, pad=10)
        ax.set_ylabel("QRC observable\nand prediction")
    for label, ax in zip("abc", axes):
        axis_style(ax, label)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.96, 0.915),
               ncol=2, handlelength=2, borderpad=0.55, columnspacing=1.5)
    axes[-1].set_xlabel("Sample within the held-out excerpt")
    axes[-1].set_xticks([0, 32, 64, 96, 127])
    fig.text(0.13, 0.083, "Measured laser inputs are scaled to 33 levels; quantum outputs are simulated.", fontsize=9, color=MUTED)
    fig.text(0.13, 0.059, "Error = test RMSE / training-output SD. Output panels use separate y scales.", fontsize=9, color=MUTED)
    fig.text(0.13, 0.035, "Fixed first seed (917000); all 12 seeds are retained in the accompanying results.", fontsize=9, color=MUTED)
    return save(fig, output, "laser_time_series")


def paired_plot(source: Path, output: Path, summary: dict) -> dict:
    cfg, seed = Config(), Config().seeds[0]
    x = np.arange(1, 49)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.4), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.22, top=0.79, hspace=0.37)
    fig.text(0.13, 0.956, "Same recent inputs, different earlier histories", fontsize=15)
    fig.text(0.13, 0.916, "Controlled replay of measured laser inputs  |  four-input window", fontsize=11, color=MUTED)
    for ax, rate, title, label in zip(axes, cfg.rates,
                                      ("Fast memory loss", "Slow memory loss"), "ab"):
        with np.load(source / f"seed{seed}_gamma{rate:g}.npz") as rows:
            ax.plot(x, rows["pair_y_a"][:48], color=INK, lw=1.4, marker=".", markersize=3)
            ax.plot(x, rows["pair_y_b"][:48], color=INK, lw=1.25, ls=(0, (5, 3)), alpha=0.75)
            ax.plot(x[cfg.window - 1:], rows["pair_q_a"][cfg.window - 1:48],
                    color=CORAL, lw=0.8, marker="o", markerfacecolor="white", markersize=4, zorder=4)
        axis_style(ax, label)
        ax.axvline(cfg.window, color=MUTED, ls=(0, (4, 3)), lw=1, zorder=1)
        ax.set_title(title, loc="left", fontsize=11, pad=10)
        ax.set_ylabel("QRC observable\nand prediction")
    legend = [Line2D([], [], color=INK, lw=1.4, marker=".", label="QRC: past A"),
              Line2D([], [], color=INK, lw=1.3, ls=(0, (5, 3)), label="QRC: past B"),
              Line2D([], [], color=CORAL, marker="o", markerfacecolor="white",
                     markersize=4, lw=0.8, label="QELM: shared window")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.55, 0.879),
               ncol=3, borderpad=0.6, handlelength=2.5, columnspacing=1.2)
    axes[-1].set_xticks([1, 4, 16, 32, 48])
    axes[-1].set_xlabel("Number of shared inputs received")
    fig.text(0.13, 0.145, "Dashed vertical line: the entire four-input window is now identical.", fontsize=9.5)
    fig.text(0.13, 0.114, "One QELM prediction must serve both histories, even while QRC outputs differ.", fontsize=9.5)
    fig.text(0.13, 0.069, "Measured segments were deliberately spliced; this is not a natural repeated event.", fontsize=9, color=MUTED)
    fig.text(0.13, 0.044, "Exact-expectation simulation, fixed first seed; separate y scales. Slow loss is not permanent memory.", fontsize=9, color=MUTED)
    return save(fig, output, "laser_same_window_control")


def compact_figure(raw_excerpt: np.ndarray, trajectories: dict, errors: dict):
    """Single-column layout; all 128 displayed samples and model values are unchanged.

    The author's reference is Fig. 2, p. 3 of arXiv:2608.25511v1: vertically
    aligned panels, in-panel headings and a shared framed legend. Its phase
    bands have no defined analogue in this natural excerpt and are not copied.
    """
    if len(raw_excerpt) != 128:
        raise ValueError("Keep the predeclared 128-sample excerpt")
    fig = plt.figure(figsize=(PAPER_COLUMN_INCHES, PAPER_FIGURE_HEIGHT))
    axes = [fig.add_axes([0.215, bottom, 0.75, height])
            for bottom, height in ((0.79, 0.195), (0.46, 0.295), (0.115, 0.295))]
    x = np.arange(128)
    axes[0].plot(x, raw_excerpt, color=COBALT, lw=0.85)
    axes[0].set_ylim(-12, 340)
    axes[0].set_yticks([0, 128, 256])
    axes[0].set_ylabel("Laser input", labelpad=4)
    headings = ("Measured laser", "Fast memory loss", "Slow memory loss")
    for ax, rate in zip(axes[1:], Config().rates):
        y, q = trajectories[rate]
        if len(y) != len(x) or len(q) != len(x):
            raise ValueError("Each compact trajectory must retain every excerpt sample")
        ax.plot(x, y, color=INK, lw=0.9, marker=".", markersize=1.8,
                label="QRC", zorder=3)
        ax.plot(x, q, color=CORAL, lw=0.65, marker="o", markersize=2.5,
                markerfacecolor="white", markeredgewidth=0.65, markevery=3,
                label="QELM", zorder=4)
        lo, hi = min(np.min(y), np.min(q)), max(np.max(y), np.max(q))
        span = max(float(hi - lo), 1e-4)
        ax.set_ylim(lo - 0.18 * span, hi + 0.39 * span)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_ylabel("Output", labelpad=4)
        ax.text(0.97, 0.045, f"full test: {100 * errors[rate]:.1f}%", ha="right",
                va="bottom", fontsize=7.5, color=MUTED, transform=ax.transAxes)
    for ax, letter, title in zip(axes, "abc", headings):
        ax.set_xlim(-1.5, 128.5)
        ax.set_xticks([0, 32, 64, 96, 127])
        ax.grid(axis="both", color=GRID, lw=0.45, ls=(0, (4, 4)))
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=0.65, pad=2)
        ax.text(0.028, 0.952, letter, transform=ax.transAxes,
                ha="left", va="top", fontsize=8.8, weight="bold")
        ax.text(0.12, 0.952, title, transform=ax.transAxes,
                ha="left", va="top", fontsize=8.1)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Time step in test excerpt", labelpad=4)
    handles, labels = axes[1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.602, 0.435),
                        ncol=2, fontsize=8.5, handlelength=1.5, handletextpad=0.5,
                        borderpad=0.42, columnspacing=1.2, fancybox=True,
                        framealpha=1.0, edgecolor=INK)
    legend.get_frame().set_linewidth(0.75)
    return fig, axes


def paper_style_plot(source: Path, output: Path, summary: dict, raw: np.ndarray) -> dict:
    cfg, seed = Config(), Config().seeds[0]
    excerpt = slice(cfg.washout, cfg.washout + 128)
    offset = sum(cfg.block_lengths[:2])
    trajectories, errors = {}, {}
    for rate in cfg.rates:
        with np.load(source / f"seed{seed}_gamma{rate:g}.npz") as rows:
            trajectories[rate] = (rows["natural_target"][excerpt].copy(),
                                  rows["natural_prediction"][excerpt].copy())
        errors[rate] = next(r["test_nrmse_training_sd"] for r in summary["results"]
                            if r["seed"] == seed and r["gamma"] == rate)
    with plt.rc_context({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
                         "font.size": 8.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
                         "axes.linewidth": 0.65}):
        fig, _ = compact_figure(raw[offset + cfg.washout:offset + cfg.washout + 128],
                                trajectories, errors)
        files = save(fig, output, "laser_paper_style")
    caption = (
        "Measured inputs, simulated quantum outputs. a Santa Fe laser intensity. "
        "b,c A four-qubit QELM emulates a four-qubit QRC using the latest four "
        "inputs. It tracks fast forgetting closely; deviations are larger with "
        "slow forgetting. Both use 48 exact-expectation features. The first "
        "predeclared realization is shown. Annotations give full-test NRMSE "
        "relative to training-output SD. Output scales differ; inputs are mapped "
        "to 33 levels. This is output emulation, not laser forecasting.\n"
    )
    caption_path = output / "laser_paper_style_caption.txt"
    caption_path.write_text(caption)
    files[caption_path.name] = digest(caption_path)
    return files


def render(source: Path, output: Path, data: Path) -> dict:
    summary = json.loads((source / "summary.json").read_text())
    validate_rows(source, summary)
    raw = load_laser(data)
    output.mkdir(parents=True, exist_ok=True)
    style()
    figures = natural_plot(source, output, summary, raw)
    figures.update(paired_plot(source, output, summary))
    figures.update(paper_style_plot(source, output, summary, raw))
    core_files = [ROOT / f"src/qrc_vs_qelm/{relative}" for relative in (
        "core/reservoirs.py", "core/liouvillian.py", "core/dissipators.py", "core/operators.py",
        "core/readout.py", "models/qelm.py", "models/api.py", "matching.py", "pipeline.py")]
    manifest = {
        "summary_sha256": digest(source / "summary.json"),
        "renderer_sha256": digest(Path(__file__)),
        "data_sha256": digest(data), "figures_sha256": figures,
        "dependencies": {str(p.relative_to(ROOT)): digest(p) for p in core_files},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "numerical_binding_check": "all_24_cells_recomputed_from_saved_arrays",
        "displayed_seed": Config().seeds[0],
        "natural_excerpt_test_indices": [128, 256],
        "pair_excerpt_common_input_indices": [0, 48],
        "compact_layout": {
            "width_inches": PAPER_COLUMN_INCHES,
            "height_inches": PAPER_FIGURE_HEIGHT,
            "matches_icaart_column_width": True,
            "reference": "author-supplied arXiv:2608.25511v1, Figure 2, page 3",
            "phase_bands": "omitted: no corresponding defined phases in the excerpt",
            "data_or_excerpt_selection_changed": False,
        },
        "paper_modified": False,
    }
    dump(output / "figure_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/santa_fe_laser/SF_Acont.dat")
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.output, args.data), indent=2))


if __name__ == "__main__":
    main()
