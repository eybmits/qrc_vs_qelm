#!/usr/bin/env python3
"""Render dissipation--window scaling at PRA column scale."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "paper"
    / "figures"
    / "canonical_emulation"
    / "fig07_dissipation_window_scaling_source.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "figures" / "compact_pra"
NAVY = "#2F5D8C"
AMBER = "#D97706"
TEAL = "#2E7D5B"
INK = "#1F2933"
GRID = "#D9E1E8"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "root_id",
        "teacher_n_qubits",
        "teacher_gamma",
        "teacher_dt",
        "window",
        "scaling_x",
        "history_nrmse",
        "root_fit_intercept",
        "root_fit_slope",
        "root_fit_r_squared",
        "root_fit_passed",
        "family_median_slope",
        "family_median_r_squared",
        "family_successful_roots",
        "family_n_roots",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"scaling source lacks columns: {sorted(missing)}")
    expected = {
        (n, root, gamma, window)
        for n in (2, 3)
        for root in [f"confirm{index:02d}" for index in range(1, 13)]
        for gamma in (0.2, 0.5, 1.0, 2.0, 4.0)
        for window in (1, 2, 4, 6)
    }
    observed = {
        (
            int(row.teacher_n_qubits),
            str(row.root_id),
            float(row.teacher_gamma),
            int(row.window),
        )
        for row in frame.itertuples(index=False)
    }
    if observed != expected:
        raise ValueError("scaling source does not match the 2x12x5x4 grid")
    if not np.allclose(
        frame["scaling_x"],
        frame["teacher_gamma"] * frame["teacher_dt"] * frame["window"],
    ):
        raise ValueError("scaling_x disagrees with gamma*dt*d")
    return frame.sort_values(
        ["teacher_n_qubits", "root_id", "teacher_gamma", "window"]
    ).reset_index(drop=True)


def render(source_path: Path, output_dir: Path) -> dict[str, str]:
    frame = _validated_source(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.7,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.50), sharex=True, sharey=True)
    x_line = np.linspace(0.0, 12.0, 200)
    family_colors = {2: NAVY, 3: TEAL}

    for panel, (ax, n_qubits) in enumerate(zip(axes, (2, 3))):
        family = frame[frame["teacher_n_qubits"].eq(n_qubits)]
        color = family_colors[n_qubits]
        ax.axhspan(3e-4, 0.10, color=TEAL, alpha=0.045, zorder=0)
        for _, root in family.groupby("root_id"):
            ax.scatter(
                root["scaling_x"],
                root["history_nrmse"],
                s=8,
                facecolor="white",
                edgecolor=color,
                linewidth=0.45,
                alpha=0.25,
                zorder=1,
            )
            intercept = float(root["root_fit_intercept"].iloc[0])
            slope = float(root["root_fit_slope"].iloc[0])
            ax.plot(
                x_line,
                np.exp(intercept + slope * x_line),
                color=color,
                alpha=0.18,
                linewidth=0.75,
                zorder=1,
            )

        medians = (
            family.groupby("scaling_x", as_index=False)["history_nrmse"]
            .median()
            .sort_values("scaling_x")
        )
        ax.scatter(
            medians["scaling_x"],
            medians["history_nrmse"],
            marker="D",
            s=25,
            color=AMBER,
            edgecolor=INK,
            linewidth=0.45,
            zorder=4,
            label="family median",
        )
        root_fits = family.drop_duplicates("root_id")
        median_intercept = float(root_fits["root_fit_intercept"].median())
        median_slope = float(root_fits["root_fit_slope"].median())
        median_r2 = float(root_fits["root_fit_r_squared"].median())
        ax.plot(
            x_line,
            np.exp(median_intercept + median_slope * x_line),
            color=INK,
            linewidth=2.2,
            zorder=3,
            label="median root fit",
        )
        ax.text(
            0.96,
            0.94,
            rf"slope $={median_slope:.3f}$" "\n" rf"median $R^2={median_r2:.3f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.0,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.8,
                "alpha": 0.96,
            },
        )
        ax.axhline(0.10, color="#66717C", linestyle=(0, (4, 3)), linewidth=0.9)
        ax.text(
            0.25,
            0.112,
            r"target tolerance $\tau=0.10$",
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#505A64",
        )
        ax.text(
            0.98,
            0.08,
            r"history NRMSE $\leq\tau$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.1,
            color=TEAL,
            fontweight="bold",
        )
        ax.set_yscale("log")
        ax.set_xlim(0.0, 12.35)
        ax.set_ylim(3e-4, 1.25)
        ax.set_title(
            rf"$\mathbf{{{chr(97 + panel)}}}$  $N_T={n_qubits}$",
            loc="left",
            pad=4,
        )
        ax.grid(axis="y", which="major", color=GRID, linewidth=0.60, alpha=0.85)
        ax.grid(axis="y", which="minor", color=GRID, linewidth=0.35, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("history NRMSE")
    fig.supxlabel(
        r"dissipation--window product $x=\gamma_T\,dt\,d$",
        x=0.5,
        y=0.115,
        fontsize=8.2,
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.035),
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.94, bottom=0.26, wspace=0.12)

    pdf = output_dir / "fig03_dissipation_window_scaling_compact.pdf"
    png = output_dir / "fig03_dissipation_window_scaling_compact.png"
    source = output_dir / "fig03_dissipation_window_scaling_compact_source.csv"
    metadata = {
        "Title": "Dissipation-window scaling of QRC observable history",
        "Creator": "qrc_vs_qelm compact scaling renderer",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, bbox_inches="tight", metadata=metadata)
    fig.savefig(png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    frame.to_csv(source, index=False)
    manifest = {path.name: _sha256(path) for path in (pdf, png, source)}
    ledger_path = output_dir / "dark_sector_figure_sha256.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {}
    if not isinstance(ledger, dict):
        raise ValueError("compact figure hash ledger is not a JSON object")
    ledger.update(manifest)
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render(args.source.resolve(), args.output_dir.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
