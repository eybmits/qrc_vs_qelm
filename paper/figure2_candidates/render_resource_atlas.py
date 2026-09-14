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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import paper_palette as colors
import active_paper_palette as roles
import render_information_interface_figures as source

SIZE = (7.055, 3.05)
OUT = Path(__file__).resolve().parent
STEM = "figure2_resource_atlas"
BUDGETS = (18, 24, 30, 36, 54, 72)
SHOTS = (300, 1200, 4800, 19200)
RATES = (0., .2, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3., 4.)
DISPLAY_RATES = tuple(g for g in RATES if g >= 1.)
CAPTION = (
    "Three resources, three different questions. "
    "a Blue cells mark windows that meet the sampled 10% old-history-error "
    "criterion in each of the two-, three-, and four-qubit cohorts "
    "(at least 10 of 12 realizations per size). Gray cells do not meet it; "
    "loss-rate rows are the tested settings, not equally spaced physical values. "
    "The display starts at loss rate 1, with only integer rates labeled. "
    "The omitted rates 0, 0.2, 0.5 and 0.75 fail this sampled criterion "
    "for all tested windows; all source data are retained. "
    "b More features reduce prediction error, with little improvement beyond "
    "36 here. Lines show medians and bands the 10th–90th percentiles across "
    "12 realizations per size. c Test error falls as the measurement budget grows. "
    "Points show medians; whiskers show 10th–90th root percentiles. Points are "
    "offset within each shot tier for legibility; connecting lines are guides. "
    "The dashed horizontal line is the 10% error reference, not a success verdict. The "
    "lower percentages show how often each feature count is chosen before testing: "
    "192 validation choices per shot tier (12 realizations with 16 repetitions each), "
    "rounded to whole percentages. Bold cells mark the most frequent count. "
    "All panels use separate studies; "
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
    fig.text(x / SIZE[0], 2.87 / SIZE[1], letter, weight="bold", fontsize=10)
    fig.text((x + .17) / SIZE[0], 2.87 / SIZE[1], title, fontsize=9.2)


def build_figure(history, features, shots):
    passed, f, med, low, high, share = summaries(history, features, shots)
    fig = plt.figure(figsize=SIZE)
    heading(fig, .42, "a", "History requirement")
    heading(fig, 2.75, "b", "Feature capacity")
    heading(fig, 4.82, "c", "Measurement budget")

    ax = axis(fig, .48, .82, 1.72, 1.84)
    displayed = passed.loc[list(DISPLAY_RATES)]
    ax.imshow(displayed.to_numpy(), origin="lower", aspect="auto", interpolation="none",
              cmap=ListedColormap([colors.NEUTRAL, colors.COBALT]), vmin=0, vmax=1)
    ax.set_xticks(range(6), range(1, 7))
    labeled_rates = (1., 2., 3., 4.)
    ax.set_yticks([DISPLAY_RATES.index(g) for g in labeled_rates],
                 [f"{g:g}" for g in labeled_rates])
    ax.set_xticks(np.arange(-.5, 6), minor=True)
    ax.set_yticks(np.arange(-.5, len(DISPLAY_RATES)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Inputs per window", labelpad=5)
    fig.text(.07 / SIZE[0], 1.74 / SIZE[1], "Loss rate (tested)",
             rotation=90, ha="center", va="center", fontsize=8.5)
    # Boundary is restricted to observed adjacent cells: no interpolation.
    source._draw_exact_threshold_boundary(ax, displayed.to_numpy())
    for line in ax.lines:
        line.set_linewidth(2.1 if line.get_color() == colors.INK else .8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = ax.legend(handles=[Patch(facecolor=colors.COBALT, label="10% criterion met"),
                              Patch(facecolor=colors.NEUTRAL, edgecolor=colors.GRID,
                                    label="criterion not met")],
                       loc="upper left", bbox_to_anchor=(-.03, -.25), frameon=False,
                       borderaxespad=0, fontsize=8, handlelength=.9, labelspacing=.25)

    for i, n in enumerate((2, 3, 4)):
        ax = axis(fig, 2.75, 2.10 - i * .67, 1.59, .56)
        group = f.loc[n].reindex(BUDGETS)
        x = np.array(BUDGETS)
        color = roles.QELM_FOR_TARGET_SIZES[n]
        ax.fill_between(x, group.low * 100, group.high * 100, color=color, alpha=.15, lw=0)
        ax.plot(x, group["median"] * 100, color=color, marker={2:"o",3:"D",4:"s"}[n],
                ms=3, lw=1.05, mew=.7, mfc="white" if n == 2 else color)
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
    fig.text(2.33 / SIZE[0], 1.73 / SIZE[1], "Prediction error", rotation=90,
             ha="center", va="center", fontsize=8.5)

    ax = axis(fig, 4.90, 1.34, 2.04, 1.20)
    # Discrete shot tiers increase fourfold. Small within-tier offsets separate
    # nearly coincident estimates without changing their values or intervals.
    series = [
        ("test_nrmse_B18", "18", roles.QELM_FEATURE_BUDGETS[18], "o", -.12),
        ("test_nrmse_B36", "36", roles.QELM_FEATURE_BUDGETS[36], "s", -.04),
        ("test_nrmse_B72", "72", roles.QELM_FEATURE_BUDGETS[72], "^", .04),
        ("selected_policy_test_nrmse", "Chosen", roles.QELM_SELECTED, "D", .12),
    ]
    for key, label, color, marker, offset in series:
        x = np.arange(4) + offset
        y, lower, upper = (frame.loc[key].to_numpy() * 100 for frame in (med, low, high))
        chosen = label == "Chosen"
        ax.errorbar(x, y, yerr=np.vstack((y - lower, upper - y)),
                    color=color, marker=marker, markersize=3.4 if chosen else 3.0,
                    markerfacecolor="white", markeredgewidth=.8,
                    linewidth=1.15 if chosen else .8, linestyle="--" if chosen else "-",
                    elinewidth=.65, capsize=1.8, capthick=.65,
                    zorder=4 if chosen else 2, label=label)
    ax.axhline(10, color=colors.MUTED, lw=.8, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(-.35, 3.35)
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40], ["0", "10", "20", "30", "40"])
    ax.set_xticks(range(4))
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.set_ylabel("Test error (%)", labelpad=3)
    ax.grid(axis="y", color=colors.GRID, lw=.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=color, marker=marker, markerfacecolor="white",
                      markeredgewidth=.8, markersize=3.4, linewidth=1,
                      linestyle="--" if label == "Chosen" else "-", label=label)
               for _, label, color, marker, _ in series]
    fig.text(4.82 / SIZE[0], 2.645 / SIZE[1], "Features:", fontsize=8)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(.23, 1.02),
              ncol=4, frameon=False, fontsize=8, handlelength=1,
              handletextpad=.25, columnspacing=.6, borderaxespad=0)

    # Direct row/column labels expose all choices without decoding stacked colors.
    select = axis(fig, 4.90, .47, 2.04, .64)
    for col, shot in enumerate(SHOTS):
        dominant = int(share.loc[shot].argmax())
        for row, budget in enumerate((18, 36, 72)):
            fraction = float(share.loc[shot, f"selected_fraction_B{budget}"])
            if row == dominant:
                select.add_patch(Rectangle((col-.34, row-.43), .68, .86,
                    facecolor=colors.COBALT_LIGHT, edgecolor="none", zorder=0))
            label = select.text(col, row, f"{fraction:.0%}", ha="center", va="center",
                fontsize=8.5, color=colors.INK, weight="bold" if row == dominant else "normal")
            label.set_gid(f"selection_B{budget}_shots{shot}")
    select.set_xlim(-.35, 3.35)
    select.set_ylim(2.5, -.5)
    select.set_yticks(range(3), ["18", "36", "72"])
    select.set_xticks(range(4), ["300", "1.2k", "4.8k", "19.2k"])
    select.tick_params(length=0)
    select.set_ylabel("Features", labelpad=3)
    select.set_xlabel("Shots per input", labelpad=4)
    fig.text(4.82 / SIZE[0], 1.19 / SIZE[1], "How often was each count chosen?", fontsize=8)
    for spine in select.spines.values():
        spine.set_visible(False)
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
             Path(source.__file__), Path(colors.__file__), Path(roles.__file__), Path(__file__)]
    ledger = {"status": "author_selected_publication_asset" if publication else "preview_only_not_integrated", "caption": CAPTION,
              "size_inches": SIZE, "sources_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
              "history_criterion": "At least 10/12 roots <= 0.10 in EACH size, without pooling",
              "history_rate_axis": "Ordered categories of measured settings, not continuous. Display starts at rate 1; only integer rates are labeled at their categorical row positions.",
              "history_display_rates": list(DISPLAY_RATES),
              "history_rates_omitted_from_display_only": [g for g in RATES if g not in DISPLAY_RATES],
              "history_boundary_labels": "No in-cell numbers or lower-rate annotation. Exact cell-edge boundary retained for seven displayed rate rows. All 198 source cells remain in exports and history_map; omitted rates below 1 fail the sampled criterion for every tested window. No untested values added.",
              "feature_display": list(BUDGETS), "feature_omitted_from_display_only": [108, 144],
              "feature_summary": f.reset_index().to_dict(orient="records"),
              "shot_medians": m.to_dict(), "shot_p10": lo.to_dict(), "shot_p90": hi.to_dict(),
              "shot_uncertainty": "10th-90th root percentiles drawn as capped whiskers for all 16 medians; full root values retained",
              "shot_axis": "Four ordered shot tiers; points offset within each tier by -0.12, -0.04, 0.04 and 0.12 for 18, 36, 72 and validation-selected features. Connecting lines are visual guides, not fits.",
              "shot_reference": "Horizontal 10% NRMSE reference only; no median-based success classification",
              "semantic_colors": "Blue shades denote QELM prediction errors and QELM feature counts; marker shapes and panel labels distinguish settings. The history criterion retains its separate blue/gray status encoding.",
              "validation_choice_share": share.to_dict(),
              "validation_choice_display": "All 12 shares shown as rounded percentages in directly labeled feature-count rows and shot-tier columns. Pale blue/bold marks the most frequent count at each tier; 192 validation choices per tier (12 roots x16 repetitions), 768 total, chosen before testing.",
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
