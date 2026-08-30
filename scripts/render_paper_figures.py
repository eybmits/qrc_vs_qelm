"""Render paper-only figures from committed gate artifacts.

The confirmatory raw result directories are intentionally not committed.  This
script therefore treats ``paper/figures/stage_gate_summary.csv`` and
``paper/figures/REPORT.md`` as its inputs and never recomputes statistics.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "paper" / "figures" / "stage_gate_summary.csv"
DEFAULT_REPORT = ROOT / "paper" / "figures" / "REPORT.md"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "phase_diagram_clean.pdf"

CELL_RE = re.compile(r"family_cell=tau(?P<tau>\d+)_p(?P<degree>\d+) label=(?P<label>\w+)")


@dataclass(frozen=True)
class PhaseCell:
    tau: int
    degree: int
    delta: float
    label: str
    significant: bool

    @property
    def key(self) -> str:
        return f"tau{self.tau}_p{self.degree}"


def load_phase_cells(path: Path) -> list[PhaseCell]:
    """Load the 27 pre-registered G1a cells from the committed gate table."""

    cells: list[PhaseCell] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("stage") != "1" or row.get("gate_id") != "G1a":
                continue
            match = CELL_RE.search(row.get("notes", ""))
            if match is None:
                raise ValueError(f"could not parse G1a cell notes: {row.get('notes')!r}")
            cells.append(
                PhaseCell(
                    tau=int(match.group("tau")),
                    degree=int(match.group("degree")),
                    delta=float(row["value"]),
                    label=match.group("label"),
                    significant=row.get("passed", "").strip().lower() == "true",
                )
            )
    if len(cells) != 27:
        raise ValueError(f"expected 27 frozen phase cells, found {len(cells)}")
    return cells


def load_uninformative_cells(path: Path) -> set[str]:
    """Read the confirmatory both-models-below-0.05 cell list from REPORT.md."""

    prefix = "Uninformative excluded cells:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            cells = set(re.findall(r"`(tau\d+_p\d+)`", line))
            if not cells:
                raise ValueError("uninformative-cell report line contained no cells")
            return cells
    raise ValueError("REPORT.md does not contain the uninformative-cell list")


def render_phase_diagram(
    cells: list[PhaseCell],
    uninformative: set[str],
    output: Path,
) -> None:
    """Render a compact categorical memory-by-nonlinearity phase diagram."""

    taus = sorted({cell.tau for cell in cells})
    degrees = sorted({cell.degree for cell in cells})
    tau_index = {value: index for index, value in enumerate(taus)}
    degree_index = {value: index for index, value in enumerate(degrees)}
    matrix = np.full((len(degrees), len(taus)), np.nan)
    by_position: dict[tuple[int, int], PhaseCell] = {}
    for cell in cells:
        row = degree_index[cell.degree]
        col = tau_index[cell.tau]
        matrix[row, col] = cell.delta
        by_position[(row, col)] = cell

    limit = max(float(np.nanmax(np.abs(matrix))), 0.02)
    fig, ax = plt.subplots(figsize=(8.7, 3.9), constrained_layout=True)
    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
        origin="lower",
        aspect="auto",
    )

    for (row, col), cell in by_position.items():
        is_uninformative = cell.key in uninformative
        if is_uninformative:
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor="#e5e7eb",
                    edgecolor="#6b7280",
                    hatch="xx",
                    linewidth=0.7,
                    zorder=2,
                )
            )
        elif cell.significant and cell.label in {"QRC", "QELM"} and abs(cell.delta) > 0.02:
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#111827",
                    linewidth=1.5,
                    zorder=3,
                )
            )
        text_color = "#111827" if is_uninformative or abs(cell.delta) < 0.38 else "white"
        ax.text(
            col,
            row,
            f"{cell.delta:+.2f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color,
            fontweight="semibold" if not is_uninformative else "normal",
            zorder=4,
        )

    ax.set_xticks(range(len(taus)), labels=[str(value) for value in taus])
    ax.set_yticks(range(len(degrees)), labels=[str(value) for value in degrees])
    ax.set_xlabel(r"Memory demand: delay $\tau$")
    ax.set_ylabel(r"Nonlinearity: degree $p$")
    ax.set_title("Matched QRC--QELM advantage across memory and nonlinearity demands")
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(taus), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(degrees), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, pad=0.018, shrink=0.92)
    colorbar.set_label(r"Mean test-capacity difference $\Delta C=C_{\rm QRC}-C_{\rm QELM}$")
    legend = [
        Patch(facecolor="none", edgecolor="#111827", linewidth=1.5, label="significant, informative"),
        Patch(facecolor="#e5e7eb", edgecolor="#6b7280", hatch="xx", label="both models $<0.05$"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=8, ncol=2)
    fig.text(
        0.5,
        -0.015,
        "Pre-registered informative set: QRC-favored 46.7%, QELM-favored 33.3%, ties 20.0% (15 cells).",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if output.suffix.lower() != ".png":
        fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    render_phase_diagram(
        load_phase_cells(args.summary),
        load_uninformative_cells(args.report),
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
