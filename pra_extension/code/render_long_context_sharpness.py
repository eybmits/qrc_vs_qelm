#!/usr/bin/env python3
"""Combine the unchanged N5 evidence and a separate analytic construction.

Chart contract: a compact 2x2 publication figure, 7.055 by 3.95 inches.
Panels a/b retain all 360 N5 root-series rows and show median NRMSE with
10th--90th percentile whiskers across five windows. Panel c retains the
72 N5 feature-budget rows at d=10 as dodged discrete error bars. Panel d
shows nine exact-expectation evaluations of a specified Bernstein readout
against its known information floor, not a capacity-optimal envelope.
The comparison asks how fitted finite resources and a known construction
relate to error, without pooling cohorts, equating metrics, or refitting.
Rendering: deterministic static Matplotlib PDF/PNG for the ICAART paper.
Palette: existing Cobalt/coral/violet with charcoal references; markers,
line styles, and open fills preserve non-color distinctions. Both-axis
grids, boxed legends, and the existing physical Helvetica/CM math typography.
No extra title or branding; manuscript caption supplies experiment scope.
QA: inspect the vector PDF raster and its final manuscript placement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

import paper_palette as colors
from render_information_interface_figures import (
    COMPARISON_LEGEND, _base_style, _style_legend, _style_quant_axis,
)
from render_long_context_n5 import (
    DEFAULT_ANALYSIS, DEFAULT_RESULTS, FEATURE_BUDGETS, FEATURE_SERIES, SERIES,
    _validate_feature_source, validated_feature_source, validated_source,
)
from sharpness_benchmarks import convergence_rows

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "long_context_sharpness"
STEM = "fig04_long_context_sharpness"
FIGURE_SIZE = (7.055, 3.95)
COPIES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _axes(fig: plt.Figure) -> list[plt.Axes]:
    width, height = FIGURE_SIZE
    axes = [
        fig.add_axes([left / width, bottom / height, 2.52 / width, 1.30 / height])
        for bottom in (2.36, 0.50) for left in (0.86, 4.30)
    ]
    for letter, ax in zip("abcd", axes, strict=True):
        _style_quant_axis(ax)
        ax.text(.035, .96, letter, transform=ax.transAxes, ha="left", va="top",
                fontsize=10.5, fontweight="bold")
    return axes


def _window_panels(source: pd.DataFrame, axes: list[plt.Axes]) -> None:
    """Retain the original renderer's source selection and root aggregation."""
    summary = source.groupby(["teacher_gamma", "series", "window"])["nrmse"].agg(
        median="median", lower=lambda s: s.quantile(.1), upper=lambda s: s.quantile(.9)
    )
    lower = 10.0 ** np.floor(np.log10(summary["lower"].min()))
    # A little extra log-axis headroom separates panel letters from the d=2
    # markers in this shallower layout; all data and both scales stay matched.
    upper = max(5.0, 5.0 * float(summary["upper"].max()))
    for gamma, ax in zip((1.0, 4.0), axes, strict=True):
        for series, label, color, marker, linestyle in SERIES:
            rows = summary.loc[(gamma, series)]
            median = rows["median"].to_numpy(float)
            ax.errorbar(
                rows.index.to_numpy(float), median,
                yerr=np.vstack((median - rows["lower"], rows["upper"] - median)),
                color=color, marker=marker, linestyle=linestyle, linewidth=1.1,
                markersize=3.7, markerfacecolor="white" if series == "history" else color,
                markeredgewidth=.6, elinewidth=.7, capsize=1.5, capthick=.7, label=label,
            )
        ax.axhline(.1, color=colors.MUTED, linestyle=(0, (4, 3)), linewidth=.85)
        ax.set_yscale("log")
        ax.set_ylim(lower / 1.2, upper)
        ax.set_xlim(1.6, 10.4)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}%"))
        ax.set_xlabel(r"window length $d$")
        ax.set_ylabel("common-suffix error")
        ax.text(.96, .96, rf"$\gamma_T={gamma:g}$", transform=ax.transAxes,
                ha="right", va="top")
        if gamma == 1.0:
            handles = [
                Line2D([], [], color=color, marker=marker, linestyle=linestyle,
                       markerfacecolor="white" if series == "history" else color,
                       markersize=3.7, markeredgewidth=.6, linewidth=1.1, label=label)
                for series, label, color, marker, linestyle in SERIES
            ]
            _style_legend(ax.legend(handles=handles, loc="lower left", **COMPARISON_LEGEND))


def _feature_panel(feature_source: pd.DataFrame, ax: plt.Axes) -> None:
    summary = feature_source.groupby(["teacher_gamma", "feature_budget"])["total_nrmse"].agg(
        median="median", lower=lambda s: s.quantile(.1), upper=lambda s: s.quantile(.9)
    )
    for offset, (gamma, label, marker, face) in zip((-.09, .09), FEATURE_SERIES, strict=True):
        rows = summary.loc[gamma].reindex(FEATURE_BUDGETS)
        median = rows["median"].to_numpy(float)
        ax.errorbar(
            np.arange(len(FEATURE_BUDGETS)) + offset, median,
            yerr=np.vstack((median - rows["lower"], rows["upper"] - median)),
            color=colors.CORAL, marker=marker, markerfacecolor=face, linestyle="none",
            linewidth=1.1, markersize=4.2, markeredgewidth=.8, elinewidth=.7,
            capsize=1.5, capthick=.7, label=label,
        )
    ax.axhline(.1, color=colors.MUTED, linestyle=(0, (4, 3)), linewidth=.85)
    upper = max(.4, np.ceil(1.05 * float(summary["upper"].max()) / .1) * .1)
    ax.set_ylim(0, upper)
    ax.set_yticks(np.arange(0, upper + .05, .1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}%"))
    ax.set_xlim(-.45, 2.45)
    ax.set_xticks(np.arange(len(FEATURE_BUDGETS)), labels=[str(b) for b in FEATURE_BUDGETS])
    ax.set_xlabel(r"measured features $B$")
    ax.set_ylabel("QELM error")
    ax.text(.50, .96, r"$d=10$", transform=ax.transAxes, ha="center", va="top")
    handles = [
        Line2D([], [], color=colors.CORAL, marker=marker, linestyle="none",
               markerfacecolor=face, markersize=4.2, markeredgewidth=.8, label=label)
        for _, label, marker, face in FEATURE_SERIES
    ]
    _style_legend(ax.legend(handles=handles, loc="upper right",
                           bbox_to_anchor=(.98, .96), **COMPARISON_LEGEND))


def _construction_panel(rows: list[dict[str, Any]], ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlim(.8, 320)
    ax.set_ylim(.070, .355)
    ax.xaxis.set_major_locator(FixedLocator([1, 4, 16, 64, 256]))
    ax.xaxis.set_minor_locator(FixedLocator([2, 8, 32, 128]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([.1, .2, .3]))
    ax.yaxis.set_minor_locator(FixedLocator([.15, .25]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(axis="both", which="minor", color=colors.GRID, linewidth=.25,
            linestyle=(0, (4, 8)), alpha=.6)
    ax.plot([row["r"] for row in rows], [row["error"] for row in rows],
            color=colors.COBALT, marker="o", markersize=3.7, markeredgewidth=.6,
            linewidth=1.1, label="Constructed QELM", zorder=3)
    ax.axhline(rows[0]["floor"], color=colors.INK, linewidth=.95,
               linestyle=(0, (4, 3)), label="Best possible error", zorder=2)
    ax.set_xlabel(r"encoding copies per input $r$")
    ax.set_ylabel("worst-case\nabsolute error")
    ax.text(.96, .96, "constructed", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=colors.MUTED)
    _style_legend(ax.legend(loc="upper right", bbox_to_anchor=(.99, .83),
                           **COMPARISON_LEGEND))


def render(source: pd.DataFrame, output: Path = DEFAULT_OUTPUT, *,
           feature_source: pd.DataFrame) -> dict[str, Any]:
    _validate_feature_source(source, feature_source)
    rows = convergence_rows(d=8, eta=.8, beta=2.0, copies=COPIES)
    if tuple(row["r"] for row in rows) != COPIES:
        raise ValueError("Incomplete construction copy counts")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pdf, png = (output / f"{STEM}.{ext}" for ext in ("pdf", "png"))
    with plt.rc_context():
        _base_style()
        fig = plt.figure(figsize=FIGURE_SIZE)
        axes = _axes(fig)
        _window_panels(source, axes[:2])
        _feature_panel(feature_source, axes[2])
        _construction_panel(rows, axes[3])
        fig.savefig(pdf, metadata={
            "Title": "Long-context evaluation and an exact construction check",
            "Creator": "qrc_vs_qelm long-context sharpness renderer",
            "CreationDate": None, "ModDate": None,
        })
        fig.savefig(png, dpi=240, metadata={"Software": "qrc_vs_qelm"})
        plt.close(fig)
    source_paths = [Path(__file__).resolve()] + [
        Path(__file__).with_name(name) for name in (
            "render_long_context_n5.py", "render_information_interface_figures.py",
            "sharpness_benchmarks.py", "paper_palette.py",
        )
    ]
    provenance = {
        "schema_version": 1,
        "window_rows": source.to_dict(orient="records"),
        "feature_rows": feature_source.to_dict(orient="records"),
        "convergence_rows": rows,
        "panels": {
            "a": {"metric": "common-suffix NRMSE", "gamma": 1},
            "b": {"metric": "common-suffix NRMSE", "gamma": 4},
            "c": {"metric": "QELM NRMSE", "window": 10},
            "d": {"metric": "global worst-case absolute scalar error",
                  "window": 8, "eta": .8, "beta": 2.0, "readout": "specified Bernstein"},
        },
        "scope": "a-c retain the original N5 cohort; d is a separate analytic construction, not fitted",
        "figure_size_inches": list(FIGURE_SIZE), "intended_width_cm": 17.9,
        "sources": {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
        "artifacts": {pdf.name: _sha256(pdf), png.name: _sha256(png)},
    }
    (output / f"{STEM}_source.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(validated_source(args.results, args.analysis), args.out,
           feature_source=validated_feature_source(args.results, args.analysis))
    print(args.out / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
