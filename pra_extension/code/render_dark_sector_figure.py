#!/usr/bin/env python3
"""Render the architecture-level QRC--finite-window separation figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from dark_sector import (
    collective_jumps,
    lindblad_protection_defect_upper_bound,
    robust_common_suffix_gap_bound,
    singlet_projector,
)
from qrc_vs_qelm.core import liouvillian


REPO_ROOT = Path(__file__).resolve().parents[2]


BLUE = "#2F5D8C"
ORANGE = "#D97706"
GREEN = "#2E7D5B"
GRAY = "#7A7A7A"
INK = "#1F2933"
GRID = "#D9E1E8"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.4,
    )


def _resource_row(
    ax: plt.Axes,
    y: float,
    symbol: str,
    label: str,
    action: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (0.03, y),
        0.94,
        0.17,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9,
        edgecolor=color,
        facecolor=color + "12",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(0.07, y + 0.085, symbol, transform=ax.transAxes, va="center", ha="left", fontsize=8.1, color=color, fontweight="bold")
    ax.text(0.27, y + 0.085, label, transform=ax.transAxes, va="center", ha="left", fontsize=7.1, color="#202833")
    ax.text(0.93, y + 0.085, action, transform=ax.transAxes, va="center", ha="right", fontsize=6.8, color=color)


def _panel_title(ax: plt.Axes, letter: str, title: str, *, pad: float = 2.0) -> None:
    """Use one neutral panel-title grammar across diagram and data panels."""

    ax.set_title(
        rf"$\mathbf{{{letter}}}$  {title}",
        loc="left",
        pad=pad,
    )


def _generator_protection_bound(*, gamma: float, asymmetry: float, dt: float) -> float:
    """Return the rigorous bound ``dt ||L^dagger(P_- )||_infty``.

    Every configured Hamiltonian term commutes with ``P_-``, so only the
    dissipator contributes.  Adjoint-channel contractivity then makes this an
    upper bound on the one-step protection defect for every suffix input.
    """

    generator = liouvillian.lindbladian(
        np.zeros((4, 4), dtype=complex),
        collective_jumps(float(gamma), float(asymmetry)),
    )
    return lindblad_protection_defect_upper_bound(
        generator, singlet_projector(), dt
    )


def render(
    data_path: Path,
    output_dir: Path,
    gamma: float | None = None,
    leakage_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, str]:
    table = pd.read_csv(data_path)
    required = {
        "root_id",
        "family",
        "asymmetry",
        "step",
        "population_gap",
        "suffix_only_mse_floor",
    }
    if not required.issubset(table.columns):
        raise ValueError(f"trajectory table misses {sorted(required - set(table.columns))}")
    leakage_path = leakage_path or (
        REPO_ROOT
        / "pra_extension"
        / "outputs"
        / "dark_sector_validation"
        / "dark_sector_leakage.csv"
    )
    config_path = config_path or REPO_ROOT / "configs" / "dark_sector_validation.json"
    leakage_source = pd.read_csv(leakage_path)
    leakage_required = {"root_id", "asymmetry", "predicted_initial_decay"}
    if not leakage_required.issubset(leakage_source.columns):
        raise ValueError(
            f"leakage table misses {sorted(leakage_required - set(leakage_source.columns))}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_gamma = float(config["architecture"]["gamma"])
    if gamma is None:
        gamma = configured_gamma
    if not np.isclose(float(gamma), configured_gamma):
        raise ValueError("figure gamma disagrees with frozen dark-sector configuration")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.08, 4.62), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.9, 1.1))
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])

    # (a) Architecture-level separation in one line.
    ax0.set_axis_off()
    _panel_title(ax0, "a", "Same finite window, different recurrent output")
    _box(ax0, (0.01, 0.55), 0.28, 0.25, "old input\n" + r"$\mathsf{0}$ or $\mathsf{A}$", ORANGE)
    _box(ax0, (0.36, 0.55), 0.27, 0.25, "same suffix\n" + r"$u_{t-d+1:t}$", BLUE)
    _box(ax0, (0.70, 0.55), 0.28, 0.25, "different output\n" + r"$\langle P_-\rangle$", GREEN)
    for start, end in ((0.29, 0.36), (0.63, 0.70)):
        ax0.add_patch(FancyArrowPatch(
            (start, 0.675), (end, 0.675), transform=ax0.transAxes,
            arrowstyle="-|>", mutation_scale=9, linewidth=1.1, color=GRAY,
        ))
    ax0.text(
        0.02,
        0.35,
        r"$P_-=(II-XX-YY-ZZ)/4,\quad \Phi_u^\dagger(P_-)=P_-$",
        transform=ax0.transAxes,
        fontsize=8.2,
    )
    ax0.text(
        0.02,
        0.12,
        "balanced suffix-only floor: "
        + r"$\mathrm{MSE}_{\min}=(\Delta\langle P_-\rangle)^2/4$",
        transform=ax0.transAxes,
        fontsize=8.2,
        color=INK,
    )
    styles = [
        ("collective_loss", BLUE, "collective loss"),
        ("local_loss", ORANGE, "matched local loss"),
    ]
    # (b) The complete resource law in one visual grammar.
    ax1.set_axis_off()
    _panel_title(ax1, "b", "Three distinct error resources")
    ax1.text(
        0.50,
        0.84,
        r"$R_{\rm total}=H_d+R_{\rm student}+R_{\rm shot}$",
        transform=ax1.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        color=INK,
        fontweight="bold",
    )
    _resource_row(ax1, 0.56, r"$H_d$", "omitted history", r"window $d$", BLUE)
    _resource_row(ax1, 0.32, r"$R_{\rm student}$", "student approximation", "feature map", ORANGE)
    _resource_row(ax1, 0.08, r"$R_{\rm shot}$", "sampling noise", "shot count", GREEN)

    # (c) The operational QELM floor.
    _panel_title(ax2, "c", "Suffix-only MSE floor versus window length")
    plotted: dict[str, pd.DataFrame] = {}
    for family, color, label in styles:
        subset = table[table["family"].eq(family) & table["asymmetry"].eq(0.0)].copy()
        subset["floor"] = np.maximum(subset["suffix_only_mse_floor"], 1e-18)
        median = subset.groupby("step", as_index=False)["floor"].median()
        qlo = subset.groupby("step")["floor"].quantile(0.1)
        qhi = subset.groupby("step")["floor"].quantile(0.9)
        ax2.fill_between(median["step"], qlo.to_numpy(), qhi.to_numpy(), color=color, alpha=0.14)
        ax2.semilogy(median["step"], median["floor"], color=color, lw=2.0, label=label)
        plotted[family] = median
    ax2.set_xlabel("accessible suffix / window length $d$")
    ax2.set_ylabel("balanced MSE lower bound")
    ax2.set_xlim(0, int(table["step"].max()))
    ax2.set_ylim(1e-12, 1e-2)
    ax2.grid(axis="y", alpha=0.42, lw=0.5, which="major", color=GRID)
    ax2.grid(axis="x", alpha=0.32, lw=0.45, color=GRID)
    ax2.text(
        79,
        float(plotted["collective_loss"]["floor"].iloc[-1]) * 0.72,
        "collective-loss floor",
        color=BLUE,
        fontsize=6.6,
        ha="right",
        va="top",
    )
    local_label_y = max(float(plotted["local_loss"]["floor"].iloc[-1]), 2.2e-10)
    ax2.text(
        79,
        local_label_y,
        "matched local-loss floor",
        color=ORANGE,
        fontsize=6.5,
        ha="right",
        va="bottom",
    )
    ax2.text(
        0.02,
        0.04,
        "band: 10th--90th root percentiles",
        transform=ax2.transAxes,
        fontsize=5.7,
        color=GRAY,
    )
    # (d) Approximate protection: observed witness and a rigorous horizon.
    _panel_title(ax3, "d", "Observed gap and protection bound")
    dt = float(config["architecture"]["dt"])
    pulse_angle = float(config["architecture"]["pulse_angle"])
    prefix_trace_distance = 2.0 * abs(np.sin(pulse_angle / 2.0))
    gap0 = 0.5 * np.sin(pulse_angle / 2.0) ** 2
    registered_frame = (
        leakage_source.groupby("asymmetry", as_index=False)["predicted_initial_decay"]
        .median()
        .sort_values("asymmetry")
    )
    registered = registered_frame["asymmetry"].to_numpy(float)
    registered_leakage = registered_frame["predicted_initial_decay"].to_numpy(float)
    coefficient_1 = np.sqrt(2.0 / (1.0 + (1.0 + registered) ** 2))
    coefficient_2 = (1.0 + registered) * coefficient_1
    expected_registered = float(gamma) * (coefficient_2 - coefficient_1) ** 2 / 2.0
    if not np.allclose(registered_leakage, expected_registered, atol=2e-6, rtol=0.0):
        raise ValueError("registered leakage points disagree with analytic frozen curve")
    steps = np.arange(int(table["step"].max()) + 1)
    curve_specs = [
        (0.02, GREEN, r"$\eta=.02$"),
        (0.05, ORANGE, r"$\eta=.05$"),
        (0.10, GRAY, r"$\eta=.10$"),
    ]
    robust_rows: list[dict[str, float | int | str]] = []
    for eta, color, label in curve_specs:
        family = "collective_loss" if eta == 0.0 else "asymmetric_collective_loss"
        subset = table[
            table["family"].eq(family) & np.isclose(table["asymmetry"], eta)
        ].copy()
        subset["absolute_gap"] = subset["population_gap"].abs()
        grouped = subset.groupby("step")["absolute_gap"]
        median = grouped.median().reindex(steps)
        qlo = grouped.quantile(0.1).reindex(steps)
        qhi = grouped.quantile(0.9).reindex(steps)
        if median.isna().any():
            raise ValueError(f"missing trajectory rows for asymmetry {eta}")
        ax3.fill_between(steps, qlo, qhi, color=color, alpha=0.08)
        ax3.plot(steps, median, color=color, lw=1.65, label=label)
        defect_bound = _generator_protection_bound(
            gamma=float(gamma), asymmetry=eta, dt=dt
        )
        certified = np.asarray(
            [
                robust_common_suffix_gap_bound(
                    gap0, defect_bound, int(step), prefix_trace_distance
                )
                for step in steps
            ]
        )
        horizon = gap0 / (defect_bound * prefix_trace_distance)
        ax3.plot(steps, certified, color=color, lw=1.05, linestyle=(0, (3, 2)))
        ax3.vlines(
            horizon,
            0.0,
            0.008,
            color=color,
            lw=1.0,
            linestyle=":",
            alpha=0.9,
        )
        ax3.text(
            horizon,
            0.003,
            rf"$\eta={eta:.2f}$" "\n" rf"$d_c={horizon:.0f}$",
            color=color,
            fontsize=5.6,
            ha="center",
            va="bottom",
        )
    for eta in registered:
        defect_bound = _generator_protection_bound(
            gamma=float(gamma), asymmetry=float(eta), dt=dt
        )
        horizon = gap0 / (defect_bound * prefix_trace_distance)
        for step in steps:
            robust_rows.append(
                {
                    "root_id": "certificate",
                    "asymmetry": float(eta),
                    "step": int(step),
                    "observed_gap_median": float(
                        table[
                            table["family"].eq("asymmetric_collective_loss")
                            & np.isclose(table["asymmetry"], eta)
                            & table["step"].eq(step)
                        ]["population_gap"].abs().median()
                    ),
                    "protection_defect_upper_bound": defect_bound,
                    "prefix_trace_distance": prefix_trace_distance,
                    "certified_gap_lower_bound": robust_common_suffix_gap_bound(
                        gap0, defect_bound, int(step), prefix_trace_distance
                    ),
                    "certified_horizon": horizon,
                    "source_kind": "robust_generator_certificate",
                    "gamma": float(gamma),
                }
            )
    ax3.set_xlabel("common-suffix length $d$")
    ax3.set_ylabel(r"observable gap $|\Delta p_d|$")
    ax3.set_xlim(0, int(steps[-1]))
    ax3.set_ylim(0, 0.132)
    ax3.grid(axis="both", alpha=0.32, lw=0.45, color=GRID)

    pdf = output_dir / "fig01_protected_excited_visible.pdf"
    png = output_dir / "fig01_protected_excited_visible.png"
    source = output_dir / "fig01_protected_excited_visible_source.csv"
    leakage_export = output_dir / "fig01_protected_excited_visible_leakage_source.csv"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    table.to_csv(source, index=False)
    registered_export = leakage_source.copy()
    registered_export["source_kind"] = "registered_frozen_points"
    registered_export["gamma"] = float(gamma)
    eta_curve = np.linspace(0.0, 0.5, 151)
    coefficient_1 = np.sqrt(2.0 / (1.0 + (1.0 + eta_curve) ** 2))
    coefficient_2 = (1.0 + eta_curve) * coefficient_1
    leakage_curve = float(gamma) * (coefficient_2 - coefficient_1) ** 2 / 2.0
    curve_export = pd.DataFrame(
        {
            "root_id": "analytic",
            "asymmetry": eta_curve,
            "predicted_initial_decay": leakage_curve,
            "source_kind": "analytic_from_frozen_config",
            "gamma": float(gamma),
        }
    )
    robust_export = pd.DataFrame(robust_rows)
    pd.concat(
        [registered_export, curve_export, robust_export], ignore_index=True
    ).to_csv(
        leakage_export, index=False
    )
    manifest = {
        pdf.name: _sha256(pdf),
        png.name: _sha256(png),
        source.name: _sha256(source),
        leakage_export.name: _sha256(leakage_export),
    }
    ledger_path = output_dir / "dark_sector_figure_sha256.json"
    ledger: dict[str, str] = {}
    if ledger_path.is_file():
        loaded = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("compact figure hash ledger is not a JSON object")
        ledger = {str(key): str(value) for key, value in loaded.items()}
    ledger.update(manifest)
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "pra_extension" / "outputs" / "dark_sector_validation" / "dark_sector_trajectories.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper" / "figures" / "compact_pra",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = render(args.data.resolve(), args.output_dir.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
