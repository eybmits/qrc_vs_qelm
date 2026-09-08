#!/usr/bin/env python3
"""Presentation-only alternatives for Figure 2; never writes the live paper."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, PercentFormatter
import numpy as np
import pandas as pd

import render_information_interface_figures as original

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper/figures/information_interface/fig03_resource_diagnosis_source.csv"
OUTPUT = ROOT / "paper/figures/resource_variants"
COLORS = {2: "#3B86FF", 3: "#FC4D30", 4: "#2A788E"}
MARKERS = {2: "o", 3: "D", 4: "s"}
SHOT_STYLES = {
    "test_nrmse_B18": ("#73CB46", "o", "18 features"),
    "test_nrmse_B36": ("#2A788E", "s", "36 features"),
    "test_nrmse_B72": ("#482878", "D", "72 features"),
    "selected_policy_test_nrmse": ("#24292F", "*", "validation-selected"),
}
SELECTED = "selected_policy_test_nrmse"
INK, MUTED, GRID = "#202830", "#69747E", "#E1E5E9"
X = {"a": "scaling_x", "b": "feature_budget", "c": "total_shots"}
PANEL = {"a": "history", "b": "features", "c": "shots"}
LABELS = {
    "a": (r"scaled context $\gamma_T\,\Delta t\,d$", "old-history error"),
    "b": (r"measured features $B$", "QELM emulation error"),
    "c": (r"shots per input $Q$", "QELM test error"),
}
TITLES = {
    "a1": "Klar & klassisch", "a2": "Weiche Streuungsbänder", "a3": "QRC-Größen getrennt",
    "b1": "Klar & klassisch", "b2": "Übersicht mit Detailfenster", "b3": "Einzelne Roots sichtbar",
    "c1": "Klar & klassisch", "c2": "Auswahlkurve im Fokus", "c3": "Feature-Budgets getrennt",
}
PROTECTED = [
    ROOT / "paper/canonical_emulation/main.pdf",
    ROOT / "paper/icaart2027/submission.pdf",
    ROOT / "paper/figures/information_interface/fig03_resource_diagnosis.pdf",
    SOURCE,
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source() -> pd.DataFrame:
    """Revalidate export identity and the live underlying frozen inputs."""
    ledger = json.loads(SOURCE.with_name("figure_sha256.json").read_text())
    if ledger[SOURCE.name] != sha(SOURCE):
        raise ValueError("Figure 2 source does not match its hash ledger")
    original._feature_density_source(original.DEFAULT_FEATURE_DENSITY_SOURCE)
    original._resource_history_with_n4(
        original.validated_scaling_source(original.DEFAULT_SCALING_SOURCE))
    original._validated_shot_effects(original.DEFAULT_SHOT_SOURCE)
    frame = pd.read_csv(SOURCE, float_precision="round_trip")
    # This design gallery predates the intermediate-bank experiment. Preserve
    # its original six-budget comparison while validating the current export.
    if frame["feature_budget"].isin((24, 30)).any():
        original._feature_density_with_bridge(original.DEFAULT_FEATURE_DENSITY_SOURCE)
        frame = frame[~(frame["panel"].eq("features") & frame["feature_budget"].isin((24, 30)))].copy()
    if frame.groupby("panel").size().to_dict() != {"history": 504, "features": 216, "shots": 192}:
        raise ValueError("Incomplete Figure 2 export")
    if not np.isfinite(frame["value"]).all() or not frame["value"].gt(0).all():
        raise ValueError("Invalid error values")
    return frame


def summaries(frame: pd.DataFrame, panel: str) -> dict:
    data = frame[frame["panel"].eq(PANEL[panel])]
    group = "metric" if panel == "c" else "teacher_n_qubits"
    return {
        key: rows.groupby(X[panel])["value"].agg(
            median="median", q10=lambda v: v.quantile(.10), q90=lambda v: v.quantile(.90)
        ).reset_index()
        for key, rows in data.groupby(group)
    }


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 11, "axes.labelweight": "normal",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.linewidth": .7, "svg.fonttype": "none",
        "svg.hashsalt": "qrc-resource-variants", "savefig.facecolor": "white",
    })


def style_axis(ax, panel: str, *, bottom: bool = True, detail: bool = False) -> None:
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#9FA8AF")
    ax.tick_params(length=3, width=.65, pad=4, labelsize=9)
    ax.grid(which="major", axis="both", color=GRID, lw=.6)
    ax.grid(which="minor", axis="both", color=GRID, lw=.3, alpha=.42)
    if panel == "a":
        ax.set_yscale("log")
        ax.set_xlim(-.15, 12.5)
        ax.set_ylim(.0004, 1.3)
        ax.set_xticks([0, 3, 6, 9, 12])
        ax.set_yticks([.001, .01, .1, 1], ["0.1%", "1%", "10%", "100%"])
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    elif panel == "b":
        ax.set_xlim(12, 150)
        ax.set_ylim(.02, .41)
        ax.set_xticks([18, 36, 54, 72, 108, 144])
        ax.set_yticks([.05, .1, .2, .3, .4])
        ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.set_xscale("log", base=4)
        ax.set_xlim(260, 22000)
        ax.set_ylim(.06, .335)
        ax.set_xticks([300, 1200, 4800, 19200], ["300", "1.2k", "4.8k", "19.2k"])
        ax.set_yticks([.1, .2, .3])
        ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.axhline(.1, color="#828B92", lw=.9, ls=(0, (4, 3)), zorder=1)
    if not detail:
        ax.text(.98, .1, "10% target", transform=ax.get_yaxis_transform(),
                color=MUTED, ha="right", va="bottom", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    if bottom:
        ax.set_xlabel(LABELS[panel][0], labelpad=10)
    else:
        ax.tick_params(labelbottom=False)


def line(ax, s, panel, color, marker, *, band=False, interval=True, selected=False,
         alpha=1., zorder=3):
    x, y = s[X[panel]].to_numpy(), s["median"].to_numpy()
    if band:
        ax.fill_between(x, s["q10"], s["q90"], color=color, alpha=.12,
                        linewidth=0, zorder=2)
    if interval and not band:
        ax.errorbar(x, y, yerr=[y-s["q10"], s["q90"]-y], fmt="none",
                    color=color, alpha=.65*alpha, elinewidth=.85, capsize=2,
                    capthick=.8, zorder=zorder)
    return ax.plot(x, y, color=color, marker=marker, lw=1.55,
                   ms=7 if selected else 4.7, markeredgewidth=1.05,
                   markerfacecolor="white" if marker == "o" else color,
                   ls=(0, (4, 2)) if selected else "-", alpha=alpha, zorder=zorder+1)[0]


def legend(host, panel):
    if panel != "c":
        handles = [Line2D([], [], color=COLORS[n], marker=MARKERS[n], lw=1.5,
                          markerfacecolor="white" if n == 2 else COLORS[n],
                          label=f"{n}-qubit QRC", ms=4.5) for n in (2, 3, 4)]
        host.legend(handles=handles, loc="center", bbox_to_anchor=(.55, .855),
                    ncol=3, frameon=False, fontsize=9, handlelength=1.6,
                    columnspacing=1.2, handletextpad=.5)
    else:
        handles = [Line2D([], [], color=c, marker=m, label=label,
                          ls="--" if k == SELECTED else "-", lw=1.5, ms=5)
                   for k, (c, m, label) in SHOT_STYLES.items()]
        host.legend(handles=handles, loc="center", bbox_to_anchor=(.56, .846),
                    ncol=2, frameon=False, fontsize=9, columnspacing=1.6,
                    handlelength=2, handletextpad=.6, labelspacing=.6)


def draw_variant(host, frame: pd.DataFrame, panel: str, version: int):
    key = f"{panel}{version}"
    host.text(.055, .957, key, fontsize=16, fontweight="bold", va="top")
    host.text(.16, .954, TITLES[key], fontsize=13, va="top")
    data = frame[frame["panel"].eq(PANEL[panel])]
    sums = summaries(frame, panel)
    footer = ("12 roots / QRC-Größe · Median + 10.–90. Perzentil" if panel != "c"
              else "12 roots · Streuung nur für die ausgewählte Policy")
    host.text(.55, .027, footer, ha="center", fontsize=8, color=MUTED)

    if version == 3:
        host.text(.55, .873, "Gleiche Achsen in allen Teilplots", ha="center",
                  fontsize=9, color=MUTED)
        axes = []
        for index, group in enumerate((2, 3, 4)):
            ax = host.add_axes([.17, .17 + (2-index)*.217, .77, .185])
            style_axis(ax, panel, bottom=index == 2, detail=True)
            ax.tick_params(labelsize=8, pad=2)
            if panel == "c":
                budget = (18, 36, 72)[index]
                metric = f"test_nrmse_B{budget}"
                c, m, label = SHOT_STYLES[metric]
                line(ax, sums[metric], panel, c, m, interval=False)
                line(ax, sums[SELECTED], panel, INK, "*", selected=True, zorder=5)
                ax.text(.97, .86, label, transform=ax.transAxes, ha="right",
                        color=c, fontsize=9, va="top")
            else:
                selected = data[data["teacher_n_qubits"].eq(group)]
                ax.scatter(selected[X[panel]], selected["value"], s=7,
                           color=COLORS[group], alpha=.20, edgecolors="none", zorder=2)
                line(ax, sums[group], panel, COLORS[group], MARKERS[group])
                ax.text(.97, .87, f"{group}-qubit QRC", color=COLORS[group],
                        transform=ax.transAxes, ha="right", va="top", fontsize=9)
            axes.append(ax)
        host.text(.035, .5, LABELS[panel][1], rotation=90, va="center", fontsize=11)
        if panel == "c":
            host.text(.55, .81, "Schwarz gestrichelt: dieselbe ausgewählte Policy",
                      fontsize=8, color=MUTED, ha="center")
        else:
            host.text(.55, .815, "Blasse Punkte: einzelne roots", fontsize=8,
                      color=MUTED, ha="center")
        return axes

    legend(host, panel)
    ax = host.add_axes([.17, .18, .77, .595])
    style_axis(ax, panel)
    ax.set_ylabel(LABELS[panel][1], labelpad=9)
    if panel in ("a", "b"):
        for n in (2, 3, 4):
            line(ax, sums[n], panel, COLORS[n], MARKERS[n], band=key == "a2")
    else:
        for metric, (color, marker, _) in SHOT_STYLES.items():
            selected = metric == SELECTED
            line(ax, sums[metric], panel, color, marker,
                 band=version == 2 and selected, interval=selected,
                 selected=selected, alpha=.68 if version == 2 and not selected else 1.,
                 zorder=5 if selected else 3)
    if key == "b2":
        inset = ax.inset_axes([.38, .51, .58, .39])
        inset.set_facecolor("#FFFFFF")
        for n in (2, 3, 4):
            line(inset, sums[n][sums[n]["feature_budget"].ge(36)], "b", COLORS[n],
                 MARKERS[n], interval=False)
        inset.set_xlim(31, 149)
        inset.set_ylim(.043, .058)
        inset.set_xticks([36, 72, 108, 144])
        inset.set_yticks([.045, .05, .055], ["4.5%", "5.0%", "5.5%"])
        inset.grid(color=GRID, lw=.45)
        inset.tick_params(labelsize=7, length=2)
        inset.set_title(r"Detail $B\geq36$ · medians only", fontsize=8, pad=6)
        for spine in inset.spines.values():
            spine.set_color("#B8C0C6")
            spine.set_linewidth(.7)
        return [ax, inset]
    return [ax]


def render(output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(ROOT)): sha(p) for p in PROTECTED}
    frame = load_source()
    setup_style()
    files = []
    for panel in "abc":
        for version in (1, 2, 3):
            fig = plt.figure(figsize=(5.25, 5.5), facecolor="white")
            draw_variant(fig, frame, panel, version)
            for ext in ("svg", "png"):
                path = output / f"{panel}{version}.{ext}"
                fig.savefig(path, dpi=180, metadata={"Date": None} if ext == "svg" else None)
                files.append(path)
            plt.close(fig)
        fig = plt.figure(figsize=(15.75, 5.5), facecolor="white")
        for version, subfig in enumerate(fig.subfigures(1, 3, wspace=.035), 1):
            draw_variant(subfig, frame, panel, version)
        path = output / f"{panel}_selection.png"
        fig.savefig(path, dpi=160)
        files.append(path)
        plt.close(fig)
    fig = plt.figure(figsize=(15.75, 16.5), facecolor="white")
    for row, panel in zip(fig.subfigures(3, 1, hspace=.03), "abc", strict=True):
        for version, subfig in enumerate(row.subfigures(1, 3, wspace=.035), 1):
            draw_variant(subfig, frame, panel, version)
    path = output / "all_nine.png"
    fig.savefig(path, dpi=145)
    files.append(path)
    plt.close(fig)
    after = {str(p.relative_to(ROOT)): sha(p) for p in PROTECTED}
    if before != after:
        raise RuntimeError("Live paper unexpectedly changed")
    manifest = {
        "purpose": "presentation alternatives, not selected manuscript replacements",
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha(SOURCE),
        "row_counts": frame.groupby("panel").size().to_dict(),
        "protected_artifacts_unchanged": before,
        "renderer_sha256": sha(Path(__file__)),
        "artifacts": {p.name: sha(p) for p in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(render(), indent=2))
