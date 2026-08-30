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

INK = "#1F2933"
MUTED = "#5E6A75"
GRID = "#D9E1E8"
GREY = "#8A929B"
GREY_LIGHT = "#EEF1F4"
NAVY = "#2F5D8C"
NAVY_MID = "#5C83AD"
NAVY_LIGHT = "#AFC5DA"
AMBER = "#D97706"
AMBER_LIGHT = "#F5D6A8"
TEAL = "#2A7F78"
TEAL_LIGHT = "#C5E0DC"
WHITE = "#FFFFFF"


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
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    text: str,
    fontsize: float = 7.5,
    weight: str = "normal",
    text_color: str = INK,
    linewidth: float = 1.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
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
        linespacing=1.15,
    )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    style: str = "-|>",
    linewidth: float = 1.1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
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
    fig = plt.figure(figsize=(7.25, 4.25), facecolor=WHITE)
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.23)
    axes = [fig.add_subplot(grid[index]) for index in range(4)]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title(r"$\mathbf{a}$  One fixed trajectory pair", loc="left", pad=5)
    ax.text(0.02, 0.78, "trajectory 0", fontsize=7.0, color=MUTED, va="center")
    ax.text(0.02, 0.47, "trajectory 1", fontsize=7.0, color=MUTED, va="center")
    _box(ax, (0.20, 0.69), 0.18, 0.18, face=GREY_LIGHT, edge=GREY, text="older input\n$(u,q)=(u_0,0)$", fontsize=6.8)
    _box(ax, (0.20, 0.38), 0.18, 0.18, face=AMBER_LIGHT, edge=AMBER, text="older input\n$(u,q)=(u_0,1)$", fontsize=6.8)
    _box(ax, (0.48, 0.53), 0.31, 0.22, face=WHITE, edge=NAVY, text="identical later inputs\n$q=0$ in every step", fontsize=7.1)
    _box(ax, (0.84, 0.53), 0.13, 0.22, face=TEAL_LIGHT, edge=TEAL, text="different\noutputs", fontsize=7.0, weight="bold")
    for y in (0.78, 0.47):
        _arrow(ax, (0.39, y), (0.48, 0.64))
    _arrow(ax, (0.79, 0.64), (0.84, 0.64))
    ax.text(
        0.635,
        0.88,
        "ALL ALLOWED SYMMETRY-PRESERVING SUFFIXES",
        ha="center",
        va="center",
        fontsize=6.5,
        color=NAVY,
        fontweight="bold",
    )
    gap = float(summary["expected_prefix_gap"])
    floor = float(summary["expected_worst_case_absolute_floor"])
    mse = float(summary["expected_balanced_mse_floor"])
    ax.text(
        0.50,
        0.17,
        rf"output gap $={gap:.6f}$   worst-case absolute error $\geq {floor:.6f}$"
        "\n"
        rf"balanced MSE floor $\simeq {mse:.6f}$",
        ha="center",
        va="center",
        fontsize=7.2,
        color=INK,
    )

    ax = axes[1]
    ax.set_title(r"$\mathbf{b}$  Protected quantum memory", loc="left", pad=5)
    specs = (
        ("PROTECTED", "later allowed\ninputs do not erase\n$P_-$", NAVY_LIGHT, NAVY),
        ("EXCITABLE", "addressable control\nchanges\n" + r"$\langle P_-\rangle$", AMBER_LIGHT, AMBER),
        ("VISIBLE", "two-body Pauli\noutputs detect\nthe change", TEAL_LIGHT, TEAL),
    )
    xs = (0.015, 0.3575, 0.70)
    for index, ((name, detail, face, edge), x) in enumerate(zip(specs, xs)):
        _box(ax, (x, 0.49), 0.28, 0.32, face=face, edge=edge, text="", fontsize=6.3)
        ax.text(
            x + 0.14,
            0.69,
            name,
            ha="center",
            va="center",
            fontsize=6.8,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            x + 0.14,
            0.585,
            detail,
            ha="center",
            va="center",
            fontsize=5.95,
            color=INK,
            linespacing=1.12,
        )
        if index < 2:
            _arrow(ax, (x + 0.28, 0.65), (xs[index + 1], 0.65), color=INK)
    ax.text(
        0.50,
        0.22,
        r"$P_-=(II-XX-YY-ZZ)/4$ is a dynamical state component,"
        "\nnot a branch label inserted into the data.",
        ha="center",
        va="center",
        fontsize=7.1,
        color=INK,
    )

    ax = axes[2]
    ax.set_title(r"$\mathbf{c}$  Three distinct limits", loc="left", pad=5)
    limits = (
        ("History missing", "increase window or retain state", GREY_LIGHT, GREY, r"$H_d$"),
        ("Map not represented", "improve features or readout", AMBER_LIGHT, AMBER, r"$R_{\rm rep}$"),
        ("Measurement too noisy", "increase measurement resolution", TEAL_LIGHT, TEAL, r"$R_{\rm shot}$"),
    )
    ys = (0.70, 0.43, 0.16)
    for (title, action, face, edge, symbol), y in zip(limits, ys):
        _box(ax, (0.04, y), 0.23, 0.17, face=face, edge=edge, text=symbol, fontsize=8.0, weight="bold")
        _arrow(ax, (0.28, y + 0.085), (0.37, y + 0.085))
        _box(ax, (0.38, y), 0.58, 0.17, face=WHITE, edge=edge, text=f"{title}\n{action}", fontsize=7.2)
    ax.text(0.50, 0.04, "Each intervention targets a different error source.", ha="center", fontsize=7.0, color=MUTED)

    ax = axes[3]
    ax.set_title(r"$\mathbf{d}$  Two-sided approximation boundary", loc="left", pad=5)
    _box(
        ax,
        (0.20, 0.78),
        0.60,
        0.14,
        face=GREY_LIGHT,
        edge=INK,
        text=r"observable suffix diameter  $D_d$",
        fontsize=7.5,
        weight="bold",
    )
    _arrow(ax, (0.37, 0.78), (0.23, 0.62), color=GREY)
    _arrow(ax, (0.63, 0.78), (0.77, 0.62), color=NAVY)
    _box(
        ax,
        (0.02, 0.41),
        0.42,
        0.20,
        face=GREY_LIGHT,
        edge=GREY,
        text=r"protected memory" "\n" r"$D_d\geq\Delta>0$",
        fontsize=7.1,
        weight="bold",
    )
    _box(
        ax,
        (0.56, 0.41),
        0.42,
        0.20,
        face=NAVY_LIGHT,
        edge=NAVY,
        text=r"fading memory" "\n" r"$D_d\rightarrow0$",
        fontsize=7.1,
        weight="bold",
    )
    _arrow(ax, (0.23, 0.41), (0.23, 0.27), color=GREY)
    _arrow(ax, (0.77, 0.41), (0.77, 0.27), color=NAVY)
    _box(
        ax,
        (0.02, 0.04),
        0.42,
        0.22,
        face=WHITE,
        edge=GREY,
        text=r"every suffix model" "\n" r"error $\geq\Delta/2$",
        fontsize=7.0,
    )
    _box(
        ax,
        (0.56, 0.04),
        0.42,
        0.22,
        face=TEAL_LIGHT,
        edge=TEAL,
        text="growing reset QELM\nhistory + map + shots",
        fontsize=7.0,
    )

    fig.subplots_adjust(left=0.025, right=0.99, top=0.97, bottom=0.035)
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
    fig.savefig(pdf, bbox_inches="tight", metadata=_metadata("Information interface and protected quantum memory"))
    fig.savefig(png, dpi=320, bbox_inches="tight")
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
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.72), sharex=True, sharey=True)
    gammas = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)
    windows = (1, 2, 4, 6)
    state_colors = {
        "HISTORY_MISSING": GREY,
        "MAP_NOT_REPRESENTED": AMBER,
        "UNRESOLVED": TEAL,
    }
    feasible_colors = {18: NAVY_LIGHT, 36: NAVY_MID, 72: NAVY}
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
                    text_color = WHITE
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=face, edgecolor=WHITE, linewidth=1.1))
                if bool(row.any_root_disagreement):
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, hatch="//", edgecolor=(1, 1, 1, 0.55), linewidth=0.0))
                if label:
                    ax.text(x, y, label, ha="center", va="center", fontsize=7.3, color=text_color, fontweight="bold")
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 5.5)
        ax.set_xticks(range(4), windows)
        ax.set_yticks(range(6), [f"{value:g}" for value in gammas])
        ax.set_xlabel("visible window $d$")
        ax.set_title(rf"$\mathbf{{{chr(97 + panel)}}}$  $N_T={n_qubits}$", loc="left", pad=5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel(r"teacher loss rate $\gamma_T$")
    handles = [
        Patch(facecolor=GREY, label="History missing"),
        Patch(facecolor=NAVY, label="Feasible; modal minimum $B$"),
        Patch(facecolor=TEAL, label="Unresolved"),
        Patch(facecolor=WHITE, edgecolor=GREY, hatch="//", label="some $B$ nonunanimous"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.7, bbox_to_anchor=(0.5, -0.005), columnspacing=1.1, handlelength=1.3)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.94, bottom=0.245, wspace=0.13)
    pdf = output_dir / "fig02_information_frontier.pdf"
    png = output_dir / "fig02_information_frontier.png"
    source = output_dir / "fig02_information_frontier_source.csv"
    table.to_csv(source, index=False)
    fig.savefig(pdf, bbox_inches="tight", metadata=_metadata("Information and feature frontier"))
    fig.savefig(png, dpi=320, bbox_inches="tight")
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
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55))
    family_colors = {2: NAVY, 3: AMBER}
    family_markers = {2: "o", 3: "D"}

    ax = axes[0]
    root_scaling = (
        scaling.groupby(["teacher_n_qubits", "root_id", "scaling_x"], as_index=False)["history_nrmse"]
        .median()
    )
    for n_qubits in (2, 3):
        family = root_scaling[root_scaling["teacher_n_qubits"].eq(n_qubits)]
        summary = family.groupby("scaling_x")["history_nrmse"].agg(
            median="median", q10=lambda values: values.quantile(0.10), q90=lambda values: values.quantile(0.90)
        ).reset_index()
        ax.fill_between(summary["scaling_x"], summary["q10"], summary["q90"], color=family_colors[n_qubits], alpha=0.12, linewidth=0)
        ax.plot(summary["scaling_x"], summary["median"], color=family_colors[n_qubits], marker=family_markers[n_qubits], markersize=3.7, linewidth=1.5, label=rf"$N_T={n_qubits}$")
    ax.axhline(0.10, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.text(0.25, 0.115, r"tolerance $0.10$", fontsize=6.1, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlim(0, 12.2)
    ax.set_ylim(5e-4, 1.15)
    ax.set_xlabel(r"$\gamma_T\,dt\,d$")
    ax.set_ylabel("history NRMSE")
    ax.set_title(r"$\mathbf{a}$  History vs forgetting", loc="left", pad=4)
    ax.legend(frameon=False, fontsize=6.3, loc="upper right", handlelength=1.6)

    ax = axes[1]
    for n_qubits in (2, 3):
        family = features[features["teacher_n_qubits"].eq(n_qubits)]
        for _, root in family.groupby("root_id"):
            ax.plot(root["feature_budget"], root["normalized_map_mse"], color=family_colors[n_qubits], alpha=0.15, linewidth=0.65)
        summary = family.groupby("feature_budget", as_index=False)["normalized_map_mse"].median()
        ax.plot(summary["feature_budget"], summary["normalized_map_mse"], color=family_colors[n_qubits], marker=family_markers[n_qubits], markersize=4.0, linewidth=1.7, label=rf"$N_T={n_qubits}$")
    ax.set_yscale("log")
    ax.set_xticks((18, 36, 72))
    ax.set_xlabel("measured features $B$")
    ax.set_ylabel(r"normalized map MSE")
    ax.set_title(r"$\mathbf{b}$  Map error vs features", loc="left", pad=4)
    ax.text(0.04, 0.06, r"$\gamma_T=4,\ d=6$", transform=ax.transAxes, fontsize=6.2, color=MUTED)
    ax.legend(frameon=False, fontsize=6.3, loc="upper right", handlelength=1.6)

    ax = axes[2]
    shot_styles = {
        "test_nrmse_B18": (GREY, "o", "-", "$B=18$"),
        "test_nrmse_B36": (NAVY_MID, "s", "-", "$B=36$"),
        "test_nrmse_B72": (NAVY, "D", "-", "$B=72$"),
        "selected_policy_test_nrmse": (INK, "*", "--", "validation-selected"),
    }
    for column, (color, marker, linestyle, label) in shot_styles.items():
        summary = shots.groupby("total_shots")[column].agg(
            median="median", q10=lambda values: values.quantile(0.10), q90=lambda values: values.quantile(0.90)
        ).reset_index()
        ax.plot(summary["total_shots"], summary["median"], color=color, marker=marker, markersize=4.1, linewidth=1.35, linestyle=linestyle, label=label)
        if column == "selected_policy_test_nrmse":
            ax.fill_between(summary["total_shots"], summary["q10"], summary["q90"], color=INK, alpha=0.08, linewidth=0)
    ax.set_xscale("log", base=4)
    ax.set_xticks((300, 1200, 4800, 19200), ("300", "1.2k", "4.8k", "19.2k"))
    ax.set_xlabel("circuit shots per input $Q$")
    ax.set_ylabel("test NRMSE")
    ax.set_title(r"$\mathbf{c}$  Test error vs shots", loc="left", pad=4)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right", handlelength=1.6, labelspacing=0.25)

    for ax in axes:
        ax.grid(axis="y", which="major", color=GRID, linewidth=0.55, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.7)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.20, wspace=0.40)
    pdf = output_dir / "fig03_resource_diagnosis.pdf"
    png = output_dir / "fig03_resource_diagnosis.png"
    source = output_dir / "fig03_resource_diagnosis_source.csv"

    scaling_export = root_scaling.rename(columns={"history_nrmse": "value"}).assign(panel="history", metric="history_nrmse")
    feature_export = features[["root_id", "teacher_n_qubits", "feature_budget", "normalized_map_mse"]].rename(columns={"normalized_map_mse": "value"}).assign(panel="features", metric="normalized_map_mse")
    shot_long = shots.melt(id_vars=["root_id", "total_shots"], value_vars=list(shot_styles), var_name="metric", value_name="value").assign(panel="shots")
    export = pd.concat([scaling_export, feature_export, shot_long], ignore_index=True, sort=False)
    export.to_csv(source, index=False)
    fig.savefig(pdf, bbox_inches="tight", metadata=_metadata("Window, feature, and measurement interventions"))
    fig.savefig(png, dpi=320, bbox_inches="tight")
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
