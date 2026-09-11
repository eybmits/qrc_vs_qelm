"""Three resource questions, using existing verified experiments.

Run from the repository root with .venv/bin/python followed by this file.
Use --publication for the approved asset; manuscript copying remains a separate step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pra_extension/code"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import paper_palette as colors
import render_information_interface_figures as source

SIZE = (7.055, 2.85)
OUT = Path(__file__).resolve().parent
STEM = "figure2_resource_atlas"
BUDGETS = (18, 24, 30, 36, 54, 72)
SHOTS = (300, 1200, 4800, 19200)
RATES = (0., .2, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3., 4.)
CAPTION = (
    "Three resources, three different questions. "
    "a Blue cells mark windows that meet the sampled 10% old-history-error "
    "criterion in each of the two-, three-, and four-qubit cohorts "
    "(at least 10 of 12 realizations per size). Gray cells do not meet it; "
    "loss-rate rows are the tested settings, not equally spaced physical values. "
    "b More features reduce prediction error, with little improvement beyond "
    "36 here. Lines show medians and bands the 10th–90th percentiles across "
    "12 realizations per size. c Median test errors (%) for fixed feature counts "
    "and validation-selected counts; small ranges show 10th–90th percentiles. "
    "Outlines mark medians below 10% before rounding. The "
    "lower bars show how often validation selects each count across 12 "
    "realizations with 16 repetitions each. All panels use separate studies; "
    "these sampled errors are not worst-case bounds."
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    history = source._validated_dense_merged_source(
        source.DEFAULT_DENSE_PHASE_SOURCE, source.DEFAULT_DENSE_SUFFIX_SOURCE,
        source.DEFAULT_DENSE_N4_PHASE_SOURCE, source.DEFAULT_DENSE_N4_SUFFIX_SOURCE)
    features = source._feature_density_with_bridge(source.DEFAULT_FEATURE_DENSITY_SOURCE)
    shots = source._validated_shot_effects(source.DEFAULT_SHOT_SOURCE)
    selected = pd.read_csv(source.DEFAULT_SHOT_SOURCE.with_name("eiv_validation_selected_rows.csv"))
    validate_selection(shots, selected)
    return history, features, shots, selected


def validate_selection(shots, selected):
    keys = ["root_id", "total_shots", "replicate"]
    if (len(selected) != 768 or selected.duplicated(keys).any()
            or not selected.groupby(keys[:2]).size().eq(16).all()
            or not selected["selected_budget_on_validation"].eq(True).all()
            or set(selected["feature_budget"]) != {18, 36, 72}):
        raise ValueError("Expected 12 roots x 4 shot tiers x 16 validation choices")
    groups = selected.groupby(keys[:2])
    expected = shots.set_index(keys[:2]).sort_index()
    actual = groups["test_noisy_feature_nrmse"].median().sort_index()
    if not actual.index.equals(expected.index):
        raise ValueError("Selection and error cohorts differ")
    np.testing.assert_allclose(actual, expected["selected_policy_test_nrmse"], atol=1e-12, rtol=0)
    for budget in (18, 36, 72):
        frequency = selected.assign(chosen=selected.feature_budget.eq(budget)).groupby(keys[:2]).chosen.mean()
        np.testing.assert_allclose(frequency, expected[f"selected_fraction_B{budget}"], atol=1e-12, rtol=0)


def summaries(history, features, shots):
    hkeys = ["teacher_n_qubits", "teacher_gamma", "window"]
    expected = {(n, g, d) for n in (2, 3, 4) for g in RATES for d in range(1, 7)}
    if len(history) != 198 or set(map(tuple, history[hkeys].to_numpy())) != expected:
        raise ValueError("Keep all 198 history cells")
    pass_map = (history.assign(passed=history.history_sufficient_n.ge(10))
                .groupby(["teacher_gamma", "window"]).passed.all()
                .unstack("window").reindex(index=RATES, columns=range(1, 7)))
    if len(features) != 288 or not features.groupby(["teacher_n_qubits", "feature_budget"]).size().eq(12).all():
        raise ValueError("Keep the complete feature cohort")
    f = (features[features.feature_budget.isin(BUDGETS)]
         .groupby(["teacher_n_qubits", "feature_budget"]).total_nrmse
         .agg(median="median", low=lambda v: v.quantile(.1), high=lambda v: v.quantile(.9)))
    columns = [f"test_nrmse_B{b}" for b in (18, 36, 72)] + ["selected_policy_test_nrmse"]
    s = shots.groupby("total_shots")[columns]
    med = s.median().reindex(SHOTS).T
    low = s.quantile(.1).reindex(SHOTS).T
    high = s.quantile(.9).reindex(SHOTS).T
    share = shots.groupby("total_shots")[[f"selected_fraction_B{b}" for b in (18, 36, 72)]].mean().reindex(SHOTS)
    return pass_map, f, med, low, high, share


def style():
    return {"font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
            "font.size": 8.5, "axes.labelsize": 8.5, "axes.labelweight": "normal",
            "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": .6,
            "text.color": colors.INK, "axes.labelcolor": colors.INK,
            "xtick.color": colors.INK, "ytick.color": colors.INK,
            "axes.edgecolor": colors.INK, "pdf.fonttype": 42, "savefig.facecolor": "white"}


def axis(fig, x, y, w, h):
    ax = fig.add_axes([x / SIZE[0], y / SIZE[1], w / SIZE[0], h / SIZE[1]])
    ax.tick_params(length=2.5, width=.6, pad=2)
    return ax


def heading(fig, x, letter, title):
    fig.text(x / SIZE[0], 2.68 / SIZE[1], letter, weight="bold", fontsize=10)
    fig.text((x + .17) / SIZE[0], 2.68 / SIZE[1], title, fontsize=9.2)


def build_figure(history, features, shots):
    passed, f, med, low, high, share = summaries(history, features, shots)
    fig = plt.figure(figsize=SIZE)
    heading(fig, .42, "a", "History requirement")
    heading(fig, 2.62, "b", "Feature capacity")
    heading(fig, 4.82, "c", "Measurement budget")

    ax = axis(fig, .48, .78, 1.57, 1.68)
    ax.imshow(passed.to_numpy(), origin="lower", aspect="auto", interpolation="none",
              cmap=ListedColormap([colors.NEUTRAL, colors.COBALT]), vmin=0, vmax=1)
    ax.set_xticks(range(6), range(1, 7))
    ax.set_yticks(range(11), [f"{g:g}" for g in RATES])
    ax.set_xticks(np.arange(-.5, 6), minor=True)
    ax.set_yticks(np.arange(-.5, 11), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Inputs per window", labelpad=5)
    fig.text(.07 / SIZE[0], 1.62 / SIZE[1], "Loss rate (tested)",
             rotation=90, ha="center", va="center", fontsize=8.5)
    # Boundary is restricted to observed adjacent cells: no interpolation.
    source._draw_exact_threshold_boundary(ax, passed.to_numpy())
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = ax.legend(handles=[Patch(facecolor=colors.COBALT, label="10% criterion met"),
                              Patch(facecolor=colors.NEUTRAL, edgecolor=colors.GRID,
                                    label="criterion not met")],
                       loc="upper left", bbox_to_anchor=(-.03, -.27), frameon=False,
                       borderaxespad=0, fontsize=8, handlelength=.9, labelspacing=.25)

    for i, n in enumerate((2, 3, 4)):
        ax = axis(fig, 2.62, 1.90 - i * .67, 1.72, .56)
        group = f.loc[n].reindex(BUDGETS)
        x = np.array(BUDGETS)
        ax.fill_between(x, group.low * 100, group.high * 100, color=colors.QRC_SIZES[n], alpha=.15, lw=0)
        ax.plot(x, group["median"] * 100, color=colors.QRC_SIZES[n], marker={2:"o",3:"D",4:"s"}[n],
                ms=3, lw=1.05, mew=.7, mfc="white" if n == 2 else colors.QRC_SIZES[n])
        ax.axhline(10, color=colors.MUTED, lw=.7, ls=(0, (4, 3)), zorder=0)
        ax.set_xlim(16, 74)
        ax.set_ylim(0, 36)
        ax.set_yticks([10, 30], ["10%", "30%"])
        ax.set_xticks(BUDGETS)
        ax.grid(axis="x", color=colors.GRID, lw=.4, ls=(0, (4, 4)))
        ax.text(.98, .95, f"{n}-qubit QRC", transform=ax.transAxes, va="top", ha="right", fontsize=8)
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Measured features", labelpad=5)
    fig.text(2.15 / SIZE[0], 1.53 / SIZE[1], "Prediction error", rotation=90,
             ha="center", va="center", fontsize=8.5)

    ax = axis(fig, 4.90, 1.06, 2.04, 1.40)
    vals = med.to_numpy() * 100
    # Labels carry exact magnitudes. Pale cell shading only groups rows.
    for row in range(4):
        for col in range(4):
            fill = colors.COBALT_LIGHT if row == 3 else colors.NEUTRAL
            ax.add_patch(Rectangle((col - .48, row - .46), .96, .92,
                                   fc=fill, ec="none", zorder=0))
            if vals[row, col] < 10:
                ax.add_patch(Rectangle((col - .48, row - .46), .96, .92,
                                       fill=False, ec=colors.COBALT, lw=1.1, zorder=1))
            ax.text(col, row - .16, f"{vals[row,col]:.1f}", ha="center", va="center",
                    fontsize=9.5, color=colors.COBALT if vals[row,col] < 10 else colors.INK)
            ax.text(col, row + .22, f"{100*low.iloc[row,col]:.0f}–{100*high.iloc[row,col]:.0f}",
                    ha="center", va="center", fontsize=8, color=colors.INK, alpha=.8)
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(3.5, -.5)
    ax.set_yticks(range(4), ["18", "36", "72", "Selected"])
    ax.tick_params(axis="both", length=0, labelbottom=False)
    ax.set_xticks(range(4))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-.03, 1.055, "Features", transform=ax.transAxes, ha="right", fontsize=8)
    ax.text(.5, 1.055, "Median test error (%)", transform=ax.transAxes, ha="center", fontsize=8)

    select = axis(fig, 4.90, .53, 2.04, .30)
    bottom = np.zeros(4)
    for b in (18, 36, 72):
        v = share[f"selected_fraction_B{b}"].to_numpy()
        select.bar(range(4), v, bottom=bottom, width=.78, color=colors.FEATURE_BUDGETS[b],
                   edgecolor="white", linewidth=.45)
        bottom += v
    select.set_xlim(-.5, 3.5)
    select.set_ylim(0, 1)
    select.set_yticks([0, 1], ["0", "100%"])
    select.set_xticks(range(4), ["300", "1.2k", "4.8k", "19.2k"])
    select.tick_params(length=0)
    select.set_xlabel("Shots per input", labelpad=4)
    select.text(0, 1.12, "Features selected by validation", transform=select.transAxes, fontsize=8)
    for spine in select.spines.values():
        spine.set_visible(False)
    # Small direct key below the table, not a box obscuring results.
    for i, b in enumerate((18, 36, 72)):
        fig.text((5.00 + i * .64) / SIZE[0], .06 / SIZE[1], f"■ {b}",
                 color=colors.FEATURE_BUDGETS[b], fontsize=8, va="bottom")
    return fig


def render(output=OUT, *, publication=False):
    history, features, shots, selected = load_data()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = "fig03_resource_diagnosis" if publication else STEM
    with plt.rc_context(style()):
        fig = build_figure(history, features, shots)
        fig.savefig(output / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(output / f"{stem}.png", dpi=240)
        plt.close(fig)
    for name, frame in (("history", history), ("features", features), ("shots", shots),
                        ("validation_choices", selected)):
        frame.to_csv(output / f"{stem}_{name}.csv", index=False)
    p, f, m, lo, hi, share = summaries(history, features, shots)
    paths = [source.DEFAULT_DENSE_PHASE_SOURCE, source.DEFAULT_DENSE_SUFFIX_SOURCE,
             source.DEFAULT_DENSE_N4_PHASE_SOURCE, source.DEFAULT_DENSE_N4_SUFFIX_SOURCE,
             source.DEFAULT_FEATURE_DENSITY_SOURCE, source.DEFAULT_SHOT_SOURCE,
             source.DEFAULT_SHOT_SOURCE.with_name("eiv_validation_selected_rows.csv"),
             Path(source.__file__), Path(colors.__file__), Path(__file__)]
    ledger = {"status": "author_selected_publication_asset" if publication else "preview_only_not_integrated", "caption": CAPTION,
              "size_inches": SIZE, "sources_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
              "history_criterion": "At least 10/12 roots <= 0.10 in EACH size, without pooling",
              "history_rate_axis": "ordered categories of measured settings, not continuous",
              "feature_display": list(BUDGETS), "feature_omitted_from_display_only": [108, 144],
              "feature_summary": f.reset_index().to_dict(orient="records"),
              "shot_medians": m.to_dict(), "shot_p10": lo.to_dict(), "shot_p90": hi.to_dict(),
              "shot_uncertainty": "10th-90th root percentiles drawn below each median; full root values retained",
              "validation_choice_share": share.to_dict(),
              "history_map": p.reset_index().to_dict(orient="records"),
              "artifacts_sha256": {p.name: digest(p) for p in sorted(output.glob(f"{stem}*"))
                                    if p.suffix in (".png", ".pdf", ".csv")}}
    (output / f"{stem}_source.json").write_text(json.dumps(ledger, indent=2) + "\n")
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args()
    ledger = render(args.output, publication=args.publication)
    print(json.dumps({"status": ledger["status"], "output": str(args.output)}, indent=2))
