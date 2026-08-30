#!/usr/bin/env python3
"""Render the three information-interface figures used by the main paper.

The renderer is presentation-only. It validates and aggregates frozen source
tables, but it never reclassifies a root-level result or refits a model.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np
import pandas as pd

from render_compact_phase_map import _validated_source as validated_phase_source
from render_compact_scaling import _validated_source as validated_scaling_source


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DARK_SUMMARY = (
    REPO_ROOT
    / "pra_extension"
    / "outputs"
    / "dark_sector_validation"
    / "dark_sector_summary.json"
)
DEFAULT_PHASE_SOURCE = (
    REPO_ROOT
    / "paper"
    / "figures"
    / "canonical_emulation"
    / "fig04_regime_map_source.csv"
)
DEFAULT_SCALING_SOURCE = (
    REPO_ROOT
    / "paper"
    / "figures"
    / "canonical_emulation"
    / "fig07_dissipation_window_scaling_source.csv"
)
DEFAULT_SUFFIX_SOURCE = (
    REPO_ROOT
    / "results_canonical_confirmatory_v1"
    / "suffix_decomposition.csv"
)
DEFAULT_SHOT_SOURCE = (
    REPO_ROOT
    / "pra_extension"
    / "outputs"
    / "finite_shots_eiv_n4_analysis_v6_final_helpers_strict"
    / "eiv_root_effects.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "figures" / "information_interface"

STYLE: dict[str, Any] = {
    "font_family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "font_size": 8.0,
    "axes_labelsize": 8.0,
    "axes_titlesize": 8.2,
    "tick_labelsize": 6.8,
    "legend_fontsize": 6.7,
    "axes_linewidth": 0.52,
    "primary_linewidth": 0.64,
    "grid_linewidth": 0.40,
}

FIGURE_SIZES = {
    "fig1": (7.055, 4.05),
    "fig2": (7.055, 2.62),
    "fig3": (7.055, 2.55),
}

PALETTE = {
    "ink": "#000000",
    "annotation": "#7F7F7F",
    "grid": "#B0B0B0",
    "neutral": "#F1F2F2",
    "history_missing": "#A7A9AC",
    "frozen": "#3B86FF",
    "frozen_light": "#EBF2FF",
    "onset": "#00A14B",
    "onset_light": "#E6F5EC",
    "scrambled": "#FC4D30",
    "scrambled_light": "#FEECE9",
    "yellow": "#FFC943",
    "pale_yellow": "#FFDD92",
    "orange": "#FF8431",
    "purple": "#A93BFF",
    "viridis_18": "#7AD151",
    "viridis_36": "#2A788E",
    "viridis_72": "#482475",
    "family_2": "#253494",
    "family_3": "#E8590C",
    "white": "#FFFFFF",
}

INK = PALETTE["ink"]
MUTED = PALETTE["annotation"]
GRID = PALETTE["grid"]
GREY = PALETTE["history_missing"]
GREY_LIGHT = PALETTE["neutral"]
NAVY = PALETTE["frozen"]
NAVY_MID = PALETTE["viridis_36"]
NAVY_LIGHT = PALETTE["frozen_light"]
AMBER = PALETTE["orange"]
AMBER_LIGHT = PALETTE["pale_yellow"]
TEAL = PALETTE["onset"]
TEAL_LIGHT = PALETTE["onset_light"]
WHITE = PALETTE["white"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(title: str) -> dict[str, Any]:
    return {
        "Title": title,
        "Creator": "qrc_vs_qelm information-interface renderer",
        "CreationDate": None,
        "ModDate": None,
    }


def _base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": STYLE["font_family"],
            "font.size": STYLE["font_size"],
            "mathtext.fontset": "cm",
            "axes.labelsize": STYLE["axes_labelsize"],
            "axes.labelweight": "bold",
            "axes.titlesize": STYLE["axes_titlesize"],
            "axes.titleweight": "normal",
            "xtick.labelsize": STYLE["tick_labelsize"],
            "ytick.labelsize": STYLE["tick_labelsize"],
            "legend.fontsize": STYLE["legend_fontsize"],
            "text.color": INK,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.linewidth": STYLE["axes_linewidth"],
            "axes.axisbelow": True,
            "xtick.color": INK,
            "ytick.color": INK,
            "hatch.linewidth": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": WHITE,
        }
    )


def _panel_heading(
    ax: plt.Axes,
    label: str,
    title: str,
    *,
    y: float = 1.015,
    title_x: float = 0.042,
) -> None:
    """Draw the reference paper's separate bold panel letter and plain title."""
    ax.text(
        0.0,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=STYLE["axes_titlesize"],
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        title_x,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=STYLE["axes_titlesize"],
        fontweight="normal",
        clip_on=False,
    )


def _style_quant_axis(ax: plt.Axes, *, log_minor_y: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(STYLE["axes_linewidth"])
    ax.tick_params(
        which="major",
        direction="out",
        length=2.2,
        width=STYLE["axes_linewidth"],
        pad=2.0,
        colors=INK,
    )
    ax.grid(
        axis="both",
        which="major",
        color=GRID,
        linewidth=STYLE["grid_linewidth"],
        alpha=0.36,
    )
    if log_minor_y:
        ax.grid(
            axis="y",
            which="minor",
            color=GRID,
            linewidth=0.30,
            alpha=0.18,
        )


def _style_legend(legend: Any) -> None:
    frame = legend.get_frame()
    frame.set_facecolor(WHITE)
    frame.set_edgecolor(INK)
    frame.set_linewidth(0.64)
    frame.set_alpha(0.96)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    text: str,
    fontsize: float = 6.8,
    weight: str = "normal",
    text_color: str = INK,
    linewidth: float = 0.70,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight=weight,
        linespacing=1.05,
    )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    style: str = "-|>",
    linewidth: float = 0.72,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=7,
            linewidth=linewidth,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _load_dark_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "expected_prefix_gap",
        "expected_worst_case_absolute_floor",
        "expected_balanced_mse_floor",
        "suffix_steps",
        "all_primary_passed",
    }
    if not isinstance(data, dict) or required - set(data):
        raise ValueError("dark-sector summary lacks required fields")
    if data["all_primary_passed"] is not True:
        raise ValueError("dark-sector validation did not pass")
    if not np.isclose(float(data["expected_prefix_gap"]), 0.11492442353296507):
        raise ValueError("unexpected protected-memory output gap")
    if not np.isclose(
        float(data["expected_worst_case_absolute_floor"]), 0.057462211766482536
    ):
        raise ValueError("unexpected worst-case absolute floor")
    if not np.isclose(float(data["expected_balanced_mse_floor"]), 0.003301905781096084):
        raise ValueError("unexpected balanced MSE floor")
    return data


def render_figure_1(dark_summary_path: Path, output_dir: Path) -> tuple[Path, ...]:
    summary = _load_dark_summary(dark_summary_path)
    _base_style()
    fig = plt.figure(figsize=FIGURE_SIZES["fig1"], facecolor=WHITE)
    grid = fig.add_gridspec(2, 2, hspace=0.24, wspace=0.20)
    axes = [fig.add_subplot(grid[index]) for index in range(4)]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    _panel_heading(ax, "a", "One fixed trajectory pair")
    ax.text(0.015, 0.70, "trajectory 0", fontsize=6.7, color=MUTED, va="center")
    ax.text(0.015, 0.40, "trajectory 1", fontsize=6.7, color=MUTED, va="center")
    _box(
        ax,
        (0.235, 0.61),
        0.205,
        0.18,
        face=GREY_LIGHT,
        edge=GREY,
        text="older input\n$(u,q)=(u_0,0)$",
        fontsize=6.5,
    )
    _box(
        ax,
        (0.235, 0.31),
        0.205,
        0.18,
        face=AMBER_LIGHT,
        edge=AMBER,
        text="older input\n$(u,q)=(u_0,1)$",
        fontsize=6.5,
    )
    _box(
        ax,
        (0.50, 0.45),
        0.29,
        0.24,
        face=WHITE,
        edge=PALETTE["purple"],
        text="identical later inputs\n$q=0$ in every step",
        fontsize=6.7,
    )
    _box(
        ax,
        (0.84, 0.46),
        0.145,
        0.22,
        face=TEAL_LIGHT,
        edge=TEAL,
        text="different\noutputs",
        fontsize=6.7,
        weight="bold",
    )
    for y in (0.70, 0.40):
        _arrow(ax, (0.44, y), (0.50, 0.57), color=INK)
    _arrow(ax, (0.79, 0.57), (0.84, 0.57), color=INK)
    ax.text(
        0.645,
        0.855,
        "All allowed symmetry-preserving\nsuffixes",
        ha="center",
        va="center",
        fontsize=6.4,
        color=PALETTE["purple"],
        fontweight="bold",
        linespacing=1.02,
    )
    gap = float(summary["expected_prefix_gap"])
    floor = float(summary["expected_worst_case_absolute_floor"])
    mse = float(summary["expected_balanced_mse_floor"])
    ax.text(
        0.50,
        0.145,
        rf"output gap $={gap:.6f}$"
        "\n"
        rf"worst-case absolute error $\geq {floor:.6f}$"
        "\n"
        rf"balanced MSE floor $\simeq {mse:.6f}$",
        ha="center",
        va="center",
        fontsize=6.6,
        color=INK,
        linespacing=1.03,
    )

    ax = axes[1]
    _panel_heading(ax, "b", "Protected quantum memory")
    specs = (
        ("PROTECTED", "later allowed\ninputs do not erase\n$P_-$", NAVY_LIGHT, NAVY),
        ("EXCITABLE", "addressable control\nchanges\n" + r"$\langle P_-\rangle$", AMBER_LIGHT, AMBER),
        ("VISIBLE", "two-body Pauli\noutputs detect\nthe change", TEAL_LIGHT, TEAL),
    )
    xs = (0.01, 0.35, 0.69)
    for index, ((name, detail, face, edge), x) in enumerate(zip(specs, xs)):
        _box(ax, (x, 0.42), 0.29, 0.38, face=face, edge=edge, text="")
        ax.text(
            x + 0.145,
            0.685,
            name,
            ha="center",
            va="center",
            fontsize=6.7,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            x + 0.145,
            0.555,
            detail,
            ha="center",
            va="center",
            fontsize=6.4,
            color=INK,
            linespacing=1.04,
        )
        if index < 2:
            _arrow(ax, (x + 0.29, 0.61), (xs[index + 1], 0.61), color=INK)
    ax.text(
        0.50,
        0.165,
        r"The singlet population is reconstructed from"
        "\nPauli correlations $XX$, $YY$, and $ZZ$.",
        ha="center",
        va="center",
        fontsize=6.65,
        color=INK,
    )

    ax = axes[2]
    _panel_heading(ax, "c", "Three separate limits")
    limits = (
        ("History missing", "increase window or retain state", GREY_LIGHT, GREY, "1"),
        ("Features too limited", "add features or improve readout", AMBER_LIGHT, AMBER, "2"),
        ("Measurements too noisy", "increase circuit shots", TEAL_LIGHT, TEAL, "3"),
    )
    ys = (0.68, 0.41, 0.14)
    for (title, action, face, edge, symbol), y in zip(limits, ys):
        _box(
            ax,
            (0.03, y),
            0.225,
            0.17,
            face=face,
            edge=edge,
            text=symbol,
            fontsize=7.7,
            weight="bold",
        )
        _arrow(ax, (0.27, y + 0.085), (0.36, y + 0.085), color=INK)
        _box(
            ax,
            (0.37, y),
            0.59,
            0.17,
            face=WHITE,
            edge=edge,
            text=f"{title}\n{action}",
            fontsize=6.8,
        )
    ax.text(
        0.50,
        0.015,
        "Each intervention targets a different error source.",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
    )

    ax = axes[3]
    _panel_heading(ax, "d", "Can a window replace recurrence?")
    _box(
        ax,
        (0.17, 0.755),
        0.66,
        0.16,
        face=GREY_LIGHT,
        edge=INK,
        text=r"Same-window output gap  $D_d$",
        fontsize=6.9,
        weight="bold",
    )
    _arrow(ax, (0.37, 0.755), (0.23, 0.61), color=GREY)
    _arrow(ax, (0.63, 0.755), (0.77, 0.61), color=NAVY)
    _box(
        ax,
        (0.015, 0.40),
        0.43,
        0.21,
        face=GREY_LIGHT,
        edge=GREY,
        text=r"protected memory" "\n" r"gap remains: $D_d\geq\Delta$",
        fontsize=6.8,
        weight="bold",
    )
    _box(
        ax,
        (0.555, 0.40),
        0.43,
        0.21,
        face=NAVY_LIGHT,
        edge=NAVY,
        text=r"fading memory" "\n" r"gap shrinks: $D_d\rightarrow0$",
        fontsize=6.8,
        weight="bold",
    )
    _arrow(ax, (0.23, 0.40), (0.23, 0.26), color=GREY)
    _arrow(ax, (0.77, 0.40), (0.77, 0.26), color=NAVY)
    _box(
        ax,
        (0.015, 0.04),
        0.43,
        0.22,
        face=WHITE,
        edge=GREY,
        text=r"No finite window" "\n" r"can remove the floor",
        fontsize=6.7,
    )
    _box(
        ax,
        (0.555, 0.04),
        0.43,
        0.22,
        face=TEAL_LIGHT,
        edge=TEAL,
        text="A longer window can work\nthen features and\nshots matter",
        fontsize=6.55,
    )

    fig.subplots_adjust(left=0.025, right=0.992, top=0.955, bottom=0.040)
    pdf = output_dir / "fig01_information_interface.pdf"
    png = output_dir / "fig01_information_interface.png"
    source = output_dir / "fig01_information_interface_source.json"
    source_payload = {
        "balanced_mse_floor": mse,
        "fixed_pair_construction": (
            "two time-homogeneous input trajectories differ once in the addressable "
            "flag and share all later symmetry-preserving inputs"
        ),
        "output_gap": gap,
        "positive_approximation_statement": (
            "continuous fading observable memory is uniformly approximable by a "
            "growing reset Bernstein QELM; contraction supplies explicit resources"
        ),
        "suffix_scope": "all allowed symmetry-preserving suffixes",
        "suffix_steps_numerically_checked": int(summary["suffix_steps"]),
        "worst_case_absolute_error_floor": floor,
    }
    source.write_text(json.dumps(source_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fig.savefig(pdf, metadata=_metadata("Information interface and protected quantum memory"))
    fig.savefig(png, dpi=320)
    plt.close(fig)
    return pdf, png, source


def simplified_phase_table(source_path: Path) -> pd.DataFrame:
    frame = validated_phase_source(source_path)
    rows: list[dict[str, Any]] = []
    keys = ["teacher_n_qubits", "teacher_gamma", "window"]
    for key, cell in frame.groupby(keys, sort=True):
        modes = dict(zip(cell["feature_budget"].astype(int), cell["modal_regime"].astype(str)))
        if set(modes) != {18, 36, 72}:
            raise ValueError("phase cell lacks a registered feature budget")
        if all(mode == "HISTORY_IMPOSSIBLE" for mode in modes.values()):
            state = "HISTORY_MISSING"
            minimum = np.nan
        else:
            emulated = sorted(budget for budget, mode in modes.items() if mode == "EMULATED")
            if emulated:
                state = "FEASIBLE"
                minimum = float(emulated[0])
            elif any(mode == "REPRESENTATION_LIMITED" for mode in modes.values()):
                state = "MAP_NOT_REPRESENTED"
                minimum = np.nan
            else:
                state = "UNRESOLVED"
                minimum = np.nan
        rows.append(
            {
                "teacher_n_qubits": int(key[0]),
                "teacher_gamma": float(key[1]),
                "window": int(key[2]),
                "state": state,
                "minimum_feature_budget": minimum,
                "any_root_disagreement": bool((cell["agreement_n"] < cell["n_roots"]).any()),
                "modal_regimes_B18_B36_B72": ";".join(modes[budget] for budget in (18, 36, 72)),
                "minimum_agreement_n": int(cell["agreement_n"].min()),
                "n_roots": int(cell["n_roots"].iloc[0]),
                "provenance_config_sha256": str(cell["provenance_config_sha256"].iloc[0]),
                "provenance_decision": str(cell["provenance_decision"].iloc[0]),
            }
        )
    result = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    if len(result) != 48:
        raise ValueError("simplified phase table must contain 48 cells")
    return result


def render_figure_2(phase_source_path: Path, output_dir: Path) -> tuple[Path, ...]:
    table = simplified_phase_table(phase_source_path)
    _base_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["fig2"], sharex=True, sharey=True)
    gammas = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)
    windows = (1, 2, 4, 6)
    state_colors = {
        "HISTORY_MISSING": GREY_LIGHT,
        "MAP_NOT_REPRESENTED": AMBER,
        "UNRESOLVED": PALETTE["family_3"],
    }
    feasible_colors = {
        18: PALETTE["viridis_18"],
        36: PALETTE["viridis_36"],
        72: PALETTE["viridis_72"],
    }
    for panel, (ax, n_qubits) in enumerate(zip(axes, (2, 3))):
        family = table[table["teacher_n_qubits"].eq(n_qubits)]
        lookup = {
            (float(row.teacher_gamma), int(row.window)): row
            for row in family.itertuples(index=False)
        }
        for y, gamma in enumerate(gammas):
            for x, window in enumerate(windows):
                row = lookup[(gamma, window)]
                if row.state == "FEASIBLE":
                    budget = int(row.minimum_feature_budget)
                    face = feasible_colors[budget]
                    label = rf"$B={budget}$"
                    text_color = WHITE if budget >= 36 else INK
                else:
                    face = state_colors[str(row.state)]
                    label = "" if row.state == "HISTORY_MISSING" else ("map" if row.state == "MAP_NOT_REPRESENTED" else "?")
                    text_color = INK if row.state == "MAP_NOT_REPRESENTED" else WHITE
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor=face,
                        edgecolor=WHITE,
                        linewidth=0.65,
                    )
                )
                if bool(row.any_root_disagreement):
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="//",
                            edgecolor=(0.14, 0.12, 0.13, 0.45),
                            linewidth=0.0,
                        )
                    )
                if label:
                    ax.text(
                        x,
                        y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color=text_color,
                        fontweight="bold",
                    )
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 5.5)
        ax.set_xticks(range(4), windows)
        ax.set_yticks(range(6), [f"{value:g}" for value in gammas])
        ax.set_xlabel("window length $d$")
        _panel_heading(ax, chr(97 + panel), rf"$N_T={n_qubits}$", y=1.035)
        ax.tick_params(
            direction="out",
            length=2.0,
            width=STYLE["axes_linewidth"],
            pad=2.0,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(INK)
            spine.set_linewidth(STYLE["axes_linewidth"])
    axes[0].set_ylabel(r"teacher loss rate $\gamma_T$")
    handles = [
        Patch(facecolor=GREY_LIGHT, edgecolor=GREY, label="History outside window"),
        Patch(facecolor=PALETTE["viridis_72"], label="Target reached; smallest $B$"),
        Patch(facecolor=PALETTE["family_3"], label="Undecided"),
        Patch(facecolor=WHITE, edgecolor=INK, hatch="//", label="Roots disagree at some $B$"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=6.7,
        bbox_to_anchor=(0.5, 0.018),
        borderpad=0.45,
        columnspacing=0.8,
        handlelength=1.15,
        handletextpad=0.45,
    )
    _style_legend(legend)
    fig.subplots_adjust(left=0.082, right=0.992, top=0.895, bottom=0.285, wspace=0.11)
    pdf = output_dir / "fig02_information_frontier.pdf"
    png = output_dir / "fig02_information_frontier.png"
    source = output_dir / "fig02_information_frontier_source.csv"
    table.to_csv(source, index=False)
    fig.savefig(pdf, metadata=_metadata("Information and feature frontier"))
    fig.savefig(png, dpi=320)
    plt.close(fig)
    return pdf, png, source


def _validated_suffix_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "root_id",
        "teacher_n_qubits",
        "teacher_gamma",
        "window",
        "model",
        "feature_budget",
        "representation_mse",
        "target_variance",
        "total_nrmse",
    }
    if missing := required - set(frame):
        raise ValueError(f"suffix source lacks columns: {sorted(missing)}")
    selected = frame[
        frame["model"].eq("ptd_parallel")
        & frame["teacher_gamma"].eq(4.0)
        & frame["window"].eq(6)
        & frame["feature_budget"].isin((18, 36, 72))
    ].copy()
    keys = ["root_id", "teacher_n_qubits", "feature_budget"]
    if selected.duplicated(keys).any() or len(selected) != 72:
        raise ValueError("feature slice must contain 12 roots x 2 families x 3 budgets")
    if (selected["target_variance"] <= 0).any():
        raise ValueError("feature slice contains nonpositive target variance")
    selected["normalized_map_mse"] = selected["representation_mse"] / selected["target_variance"]
    return selected.sort_values(keys).reset_index(drop=True)


def _validated_shot_effects(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "root_id",
        "total_shots",
        "test_nrmse_B18",
        "test_nrmse_B36",
        "test_nrmse_B72",
        "selected_policy_test_nrmse",
    }
    if missing := required - set(frame):
        raise ValueError(f"shot source lacks columns: {sorted(missing)}")
    expected_shots = {300, 1200, 4800, 19200}
    if set(frame["total_shots"].astype(int)) != expected_shots:
        raise ValueError("shot source has an unexpected total-shot grid")
    if frame["root_id"].nunique() != 12 or len(frame) != 48:
        raise ValueError("shot source must contain 12 roots x 4 shot tiers")
    if frame.duplicated(["root_id", "total_shots"]).any():
        raise ValueError("shot source contains duplicate root-shot rows")
    return frame.sort_values(["root_id", "total_shots"]).reset_index(drop=True)


def render_figure_3(
    scaling_source_path: Path,
    suffix_source_path: Path,
    shot_source_path: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    scaling = validated_scaling_source(scaling_source_path)
    features = _validated_suffix_features(suffix_source_path)
    shots = _validated_shot_effects(shot_source_path)
    _base_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["fig3"])
    family_colors = {2: PALETTE["family_2"], 3: PALETTE["family_3"]}
    family_markers = {2: "o", 3: "D"}

    ax = axes[0]
    root_scaling = (
        scaling.groupby(["teacher_n_qubits", "root_id", "scaling_x"], as_index=False)["history_nrmse"]
        .median()
    )
    for n_qubits in (2, 3):
        family = root_scaling[root_scaling["teacher_n_qubits"].eq(n_qubits)]
        for _, root in family.groupby("root_id"):
            ax.plot(
                root["scaling_x"],
                root["history_nrmse"],
                color=family_colors[n_qubits],
                alpha=0.18,
                linewidth=0.45,
                zorder=1,
            )
        summary = family.groupby("scaling_x")["history_nrmse"].agg(
            median="median", q10=lambda values: values.quantile(0.10), q90=lambda values: values.quantile(0.90)
        ).reset_index()
        ax.fill_between(
            summary["scaling_x"],
            summary["q10"],
            summary["q90"],
            color=family_colors[n_qubits],
            alpha=0.14,
            linewidth=0,
            zorder=0,
        )
        ax.plot(
            summary["scaling_x"],
            summary["median"],
            color=family_colors[n_qubits],
            marker=family_markers[n_qubits],
            markersize=3.6,
            markeredgewidth=0.52,
            linewidth=STYLE["primary_linewidth"],
            label=rf"$N_T={n_qubits}$",
            zorder=3,
        )
    ax.axhline(0.10, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.60)
    ax.text(
        11.8,
        0.115,
        r"tolerance $0.10$",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=MUTED,
    )
    ax.set_yscale("log")
    ax.set_xlim(-0.30, 12.50)
    ax.set_ylim(5e-4, 1.15)
    ax.set_xlabel(r"$\gamma_T\,dt\,d$")
    ax.set_ylabel("history NRMSE")
    _panel_heading(ax, "a", "Include enough history", y=1.035, title_x=0.070)
    legend = ax.legend(
        frameon=True,
        fontsize=6.7,
        loc="upper right",
        handlelength=1.35,
        borderpad=0.45,
        labelspacing=0.30,
    )
    _style_legend(legend)

    ax = axes[1]
    for n_qubits in (2, 3):
        family = features[features["teacher_n_qubits"].eq(n_qubits)]
        for _, root in family.groupby("root_id"):
            ax.plot(
                root["feature_budget"],
                root["total_nrmse"],
                color=family_colors[n_qubits],
                alpha=0.18,
                linewidth=0.45,
            )
        summary = family.groupby("feature_budget", as_index=False)["total_nrmse"].median()
        ax.plot(
            summary["feature_budget"],
            summary["total_nrmse"],
            color=family_colors[n_qubits],
            marker=family_markers[n_qubits],
            markersize=3.8,
            markeredgewidth=0.52,
            linewidth=STYLE["primary_linewidth"],
            label=rf"$N_T={n_qubits}$",
        )
    ax.axhline(0.10, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.60)
    ax.text(
        70.0,
        0.104,
        r"tolerance $0.10$",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=MUTED,
    )
    ax.set_ylim(0.035, 0.28)
    ax.set_xticks((18, 36, 72))
    ax.set_xlabel("measured features $B$")
    ax.set_ylabel(r"total QELM NRMSE")
    _panel_heading(ax, "b", "Add enough features", y=1.035, title_x=0.070)
    legend = ax.legend(
        frameon=True,
        fontsize=6.7,
        loc="upper right",
        handlelength=1.35,
        borderpad=0.45,
        labelspacing=0.30,
    )
    _style_legend(legend)

    ax = axes[2]
    shot_styles = {
        "test_nrmse_B18": (PALETTE["viridis_18"], "o", "-", "$B=18$"),
        "test_nrmse_B36": (PALETTE["viridis_36"], "s", "-", "$B=36$"),
        "test_nrmse_B72": (PALETTE["viridis_72"], "D", "-", "$B=72$"),
        "selected_policy_test_nrmse": (INK, "*", "--", "selected policy"),
    }
    for column, (color, marker, linestyle, label) in shot_styles.items():
        summary = shots.groupby("total_shots")[column].agg(
            median="median", q10=lambda values: values.quantile(0.10), q90=lambda values: values.quantile(0.90)
        ).reset_index()
        ax.plot(
            summary["total_shots"],
            summary["median"],
            color=color,
            marker=marker,
            markersize=3.7,
            markeredgewidth=0.52,
            linewidth=STYLE["primary_linewidth"],
            linestyle=linestyle,
            label=label,
        )
        if column == "selected_policy_test_nrmse":
            ax.fill_between(
                summary["total_shots"],
                summary["q10"],
                summary["q90"],
                color=INK,
                alpha=0.12,
                linewidth=0,
            )
    ax.set_xscale("log", base=4)
    ax.set_xticks((300, 1200, 4800, 19200), ("300", "1.2k", "4.8k", "19.2k"))
    ax.set_xlabel("circuit shots per input $Q$")
    ax.set_ylabel("test NRMSE")
    ax.axhline(0.10, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.60)
    _panel_heading(ax, "c", "Use enough shots", y=1.035, title_x=0.070)
    legend = ax.legend(
        frameon=True,
        fontsize=6.4,
        loc="upper right",
        handlelength=1.30,
        borderpad=0.50,
        labelspacing=0.25,
    )
    _style_legend(legend)

    for index, ax in enumerate(axes):
        _style_quant_axis(ax, log_minor_y=index == 0)
        ax.xaxis.labelpad = 2.5
        ax.yaxis.labelpad = 2.5
    fig.subplots_adjust(left=0.078, right=0.985, top=0.90, bottom=0.22, wspace=0.32)
    pdf = output_dir / "fig03_resource_diagnosis.pdf"
    png = output_dir / "fig03_resource_diagnosis.png"
    source = output_dir / "fig03_resource_diagnosis_source.csv"

    scaling_export = root_scaling.rename(columns={"history_nrmse": "value"}).assign(panel="history", metric="history_nrmse")
    feature_export = features[["root_id", "teacher_n_qubits", "feature_budget", "total_nrmse"]].rename(columns={"total_nrmse": "value"}).assign(panel="features", metric="total_nrmse")
    shot_long = shots.melt(id_vars=["root_id", "total_shots"], value_vars=list(shot_styles), var_name="metric", value_name="value").assign(panel="shots")
    export = pd.concat([scaling_export, feature_export, shot_long], ignore_index=True, sort=False)
    export.to_csv(source, index=False)
    fig.savefig(pdf, metadata=_metadata("Window, feature, and measurement interventions"))
    fig.savefig(png, dpi=320)
    plt.close(fig)
    return pdf, png, source


def render_all(
    *,
    dark_summary_path: Path = DEFAULT_DARK_SUMMARY,
    phase_source_path: Path = DEFAULT_PHASE_SOURCE,
    scaling_source_path: Path = DEFAULT_SCALING_SOURCE,
    suffix_source_path: Path = DEFAULT_SUFFIX_SOURCE,
    shot_source_path: Path = DEFAULT_SHOT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        *render_figure_1(dark_summary_path, output_dir),
        *render_figure_2(phase_source_path, output_dir),
        *render_figure_3(scaling_source_path, suffix_source_path, shot_source_path, output_dir),
    ]
    manifest = {path.name: _sha256(path) for path in artifacts}
    (output_dir / "figure_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dark-summary", type=Path, default=DEFAULT_DARK_SUMMARY)
    parser.add_argument("--phase-source", type=Path, default=DEFAULT_PHASE_SOURCE)
    parser.add_argument("--scaling-source", type=Path, default=DEFAULT_SCALING_SOURCE)
    parser.add_argument("--suffix-source", type=Path, default=DEFAULT_SUFFIX_SOURCE)
    parser.add_argument("--shot-source", type=Path, default=DEFAULT_SHOT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = render_all(
        dark_summary_path=args.dark_summary.resolve(),
        phase_source_path=args.phase_source.resolve(),
        scaling_source_path=args.scaling_source.resolve(),
        suffix_source_path=args.suffix_source.resolve(),
        shot_source_path=args.shot_source.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
