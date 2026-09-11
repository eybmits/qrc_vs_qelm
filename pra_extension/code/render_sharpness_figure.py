#!/usr/bin/env python3
"""Render the paper's exactly solvable midpoint-construction check.

Chart contract
--------------
Question: How does the specified Bernstein readout approach the scalar
information floor as encoding copies increase at a fixed window length?
Takeaway: This analytic construction approaches the known floor; the curve
does not describe the best readout at each capacity or learned performance.
Family: ordered line-and-marker comparison with a horizontal benchmark.
Source: nine deterministic ``sharpness_benchmarks.convergence_rows`` results,
one per tested copy count, eta=0.8, d=8, beta=2.0. No sampled observations,
interpolated data, stochastic uncertainty, or percentage normalization.
Surface: standalone Matplotlib vector PDF included at 7.5 cm in the ICAART
paper, with a PNG preview. Native footprint: 3.25 by 2.35 inches.
Palette: author's existing Cobalt root plus charcoal and gray references.
Non-color distinctions: circles and a solid construction line versus a
dashed floor. The manuscript caption supplies the model and scope context;
no extra title or decorative branding is added to this publication panel.
QA: inspect the exported PDF rendering and the final manuscript placement.
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
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

import paper_palette as colors
from sharpness_benchmarks import convergence_rows


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "sharpness"
STEM = "fig05_exact_sharpness"
FIGURE_SIZE = (3.25, 2.35)
COPIES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rc_style() -> dict[str, Any]:
    """Physical typography and Cobalt ink match the existing paper plots."""
    return {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
        "font.size": 9.5,
        "mathtext.fontset": "cm",
        "axes.labelsize": 10.0,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.55,
        "axes.axisbelow": True,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.5,
        "text.color": colors.INK,
        "axes.edgecolor": colors.INK,
        "axes.labelcolor": colors.INK,
        "xtick.color": colors.INK,
        "ytick.color": colors.INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": colors.WHITE,
        "savefig.facecolor": colors.WHITE,
    }


def render(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Write a deterministic PDF/PNG and transparent numeric provenance."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = convergence_rows(d=8, eta=0.8, beta=2.0, copies=COPIES)
    if tuple(row["r"] for row in rows) != COPIES:
        raise ValueError("Unexpected copy-count coverage in sharpness source")
    floor = rows[0]["floor"]
    if any(row["floor"] != floor or row["error"] < floor for row in rows):
        raise ValueError("Construction rows do not share a valid information floor")

    pdf = output / f"{STEM}.pdf"
    png = output / f"{STEM}.png"
    with plt.rc_context(_rc_style()):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        fig.subplots_adjust(left=0.188, right=0.975, bottom=0.225, top=0.970)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.8, 320)
        ax.set_ylim(0.070, 0.335)
        ax.xaxis.set_major_locator(FixedLocator([1, 4, 16, 64, 256]))
        ax.xaxis.set_minor_locator(FixedLocator([2, 8, 32, 128]))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.3]))
        ax.yaxis.set_minor_locator(FixedLocator([0.15, 0.25]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(which="major", direction="out", length=3.0, width=0.55, pad=3.0)
        ax.tick_params(which="minor", direction="out", length=1.8, width=0.45)
        ax.grid(
            axis="both", which="major", color=colors.GRID, linewidth=0.35,
            linestyle=(0, (2 / 0.35, 2 / 0.35)),
        )
        ax.grid(
            axis="both", which="minor", color=colors.GRID, linewidth=0.25,
            linestyle=(0, (1 / 0.25, 2 / 0.25)), alpha=0.6,
        )
        ax.plot(
            [row["r"] for row in rows], [row["error"] for row in rows],
            color=colors.COBALT, marker="o", markersize=4.0,
            markeredgewidth=0.65, linewidth=1.1, label="Constructed QELM", zorder=3,
        )
        ax.axhline(
            floor, color=colors.INK, linewidth=0.95,
            linestyle=(0, (4, 3)), label="Best possible error", zorder=2,
        )
        ax.set_xlabel(r"Encoding copies per input $r$", labelpad=5.0)
        ax.set_ylabel("Worst-case absolute error", labelpad=5.0)
        legend = ax.legend(
            loc="upper right", frameon=True, fancybox=True, handlelength=1.5,
            handletextpad=0.4, borderpad=0.35, borderaxespad=0.45, labelspacing=0.3,
        )
        frame = legend.get_frame()
        frame.set_facecolor(colors.WHITE)
        frame.set_edgecolor(colors.INK)
        frame.set_linewidth(0.64)
        frame.set_alpha(0.96)
        fig.savefig(
            pdf,
            metadata={
                "Title": "Exactly solvable QELM construction check",
                "Creator": "qrc_vs_qelm sharpness renderer",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        fig.savefig(png, dpi=240, metadata={"Software": "qrc_vs_qelm sharpness renderer"})
        plt.close(fig)

    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("sharpness_benchmarks.py"),
        Path(__file__).with_name("paper_palette.py"),
    ]
    provenance = {
        "schema_version": 1,
        "description": "Exact-expectation error of a specified Bernstein readout",
        "parameters": {"d": 8, "eta": 0.8, "beta": 2.0, "copies": list(COPIES)},
        "metric": "global worst-case absolute scalar error; not NRMSE",
        "numeric_source": "sharpness_benchmarks.convergence_rows",
        "rows": rows,
        "scope": [
            "Analytically constructed readouts, not learned or hardware results.",
            "The curve is not a capacity-optimal envelope; other coefficients can improve it.",
            "Large registers use factorized binomial expectations, not full statevectors.",
            "Lines join computed copy-count settings; no extra interpolated samples.",
        ],
        "figure_size_inches": list(FIGURE_SIZE),
        "intended_width_cm": 7.5,
        "palette": {"construction": colors.COBALT, "floor": colors.INK, "grid": colors.GRID},
        "sources": {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
        "artifacts": {pdf.name: _sha256(pdf), png.name: _sha256(png)},
    }
    (output / f"{STEM}_source.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.output_dir)
    print(args.output_dir / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
