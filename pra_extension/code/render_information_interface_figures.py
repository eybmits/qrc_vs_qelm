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
import shutil
import subprocess
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from figure1_timeseries import provenance as timeseries_provenance
import paper_palette as colors
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
DEFAULT_HISTORY_FRONTIER_SOURCE = (
    REPO_ROOT
    / "paper"
    / "figures"
    / "canonical_emulation"
    / "fig01_history_frontier_source.csv"
)
DEFAULT_DENSE_PHASE_SOURCE = (
    REPO_ROOT
    / "results_dense_phase_secondary_v1_analysis"
    / "dense_phase_cells.csv"
)
DEFAULT_DENSE_SUFFIX_SOURCE = (
    REPO_ROOT
    / "results_dense_phase_secondary_v1"
    / "suffix_decomposition.csv"
)
DEFAULT_DENSE_N4_PHASE_SOURCE = (
    REPO_ROOT
    / "results_dense_phase_n4_v1_analysis"
    / "dense_n4_phase_cells.csv"
)
DEFAULT_DENSE_N4_SUFFIX_SOURCE = (
    REPO_ROOT
    / "results_dense_phase_n4_v1"
    / "suffix_decomposition.csv"
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
DEFAULT_FEATURE_DENSITY_SOURCE = REPO_ROOT / "results_feature_density_v1/metrics.csv"
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
    "font_size": 9.5,
    "axes_labelsize": 10.0,
    "axes_titlesize": 10.5,
    "tick_labelsize": 9.0,
    "legend_fontsize": 8.5,
    "axes_linewidth": 0.55,
    "primary_linewidth": 1.1,
    "grid_linewidth": 0.35,
}

# Physical publication geometry shared by current manuscript Figures 2 and 3.
# Legacy artifact numbering is retained below for reproducibility.
COMPARISON_LAYOUT = {
    "figure_width": 7.055,
    "figure_height": 2.40,
    "panel_side": 1.60,
    "panel_gap": 0.68,
    "two_panel_gap": 0.86,
    "bottom": 0.53,
    "three_panel_left": 0.70,
}
COMPARISON_LEGEND = {
    "frameon": True,
    "fancybox": True,
    "fontsize": STYLE["legend_fontsize"],
    "handlelength": 1.2,
    "handletextpad": 0.4,
    "borderpad": 0.3,
    "borderaxespad": 0.45,
    "labelspacing": 0.25,
}

FIGURE_SIZES = {
    "fig1": (7.055, 9.60 / 2.54),
    "fig2": (7.055, 2.65),
    "fig3": (COMPARISON_LAYOUT["figure_width"], COMPARISON_LAYOUT["figure_height"]),
}

PALETTE = {
    "ink": colors.INK,
    "annotation": colors.MUTED,
    "grid": colors.GRID,
    "neutral": colors.NEUTRAL,
    "history_missing": colors.MUTED,
    "frozen": colors.COBALT,
    "frozen_light": colors.COBALT_LIGHT,
    "onset": colors.CORAL,
    "onset_light": colors.CORAL_LIGHT,
    "scrambled": colors.VIOLET,
    "scrambled_light": colors.VIOLET_LIGHT,
    "yellow": colors.FEATURE_FILL,
    "pale_yellow": colors.FEATURE_BACKGROUND,
    "orange": colors.CORAL,
    "purple": colors.VIOLET,
    "viridis_18": colors.FEATURE_BUDGETS[18],
    "viridis_36": colors.FEATURE_BUDGETS[36],
    "viridis_72": colors.FEATURE_BUDGETS[72],
    "family_2": colors.QRC_SIZES[2],
    "family_3": colors.QRC_SIZES[3],
    "story_ink": colors.INK,
    "story_border": colors.GRID,
    "story_neutral": colors.NEUTRAL,
    "story_blue": colors.COBALT,
    "story_blue_light": colors.COBALT_LIGHT,
    "story_green": colors.CORAL,
    "story_green_light": colors.CORAL_LIGHT,
    "story_orange": colors.VIOLET,
    "story_orange_light": colors.VIOLET_LIGHT,
    "white": colors.WHITE,
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
    title: str = "",
    *,
    y: float = 1.015,
    title_x: float = 0.042,
    verticalalignment: str = "bottom",
) -> None:
    """Draw a panel letter, with an optional legacy title kept separate."""
    ax.text(
        0.0,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va=verticalalignment,
        fontsize=STYLE["axes_titlesize"],
        fontweight="bold",
        clip_on=False,
    )
    if title:
        ax.text(
            title_x,
            y,
            title,
            transform=ax.transAxes,
            ha="left",
            va=verticalalignment,
            fontsize=STYLE["axes_titlesize"],
            fontweight="normal",
            clip_on=False,
        )


def _style_quant_axis(ax: plt.Axes) -> None:
    for name, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(STYLE["axes_linewidth"])
    ax.tick_params(
        which="major",
        direction="out",
        length=3.0,
        width=STYLE["axes_linewidth"],
        pad=3.0,
        colors=INK,
    )
    # Major guides in both directions match the native Figure 1 plots.
    # Keep minor ticks without a dense secondary grid; place guides below data.
    ax.set_axisbelow(True)
    ax.grid(False, axis="both", which="minor")
    ax.grid(
        axis="both",
        which="major",
        color=GRID,
        linewidth=STYLE["grid_linewidth"],
        # Matplotlib scales dash lengths by linewidth, so normalize to 2 pt.
        linestyle=(0, (2 / STYLE["grid_linewidth"],) * 2),
        alpha=1.0,
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
    """Compile the native-size TikZ explanatory graphic without rasterizing text."""

    _load_dark_summary(dark_summary_path)
    pdf = output_dir / "fig01_information_interface.pdf"
    png = output_dir / "fig01_information_interface.png"
    # Preserve the old provenance byte-for-byte: the prospective N5 seed audit
    # froze it before the current drawing revision. New renders have a separate
    # provenance file, never an updated historical audit snapshot.
    source = output_dir / "fig01_information_interface_render_source.json"
    historical_source = DEFAULT_OUTPUT / "fig01_information_interface_source.json"
    historical_sha256 = _sha256(historical_source)
    if historical_sha256 != (
        "d5a76f449661e3f4fabf909a6271208f838907f076db1c02652b771665d23965"
    ):
        raise ValueError("The frozen Figure 1 input-audit snapshot changed")
    tikz_source = DEFAULT_OUTPUT / "fig01_information_interface.tex"
    input_timeseries = timeseries_provenance(tikz_source)
    with tempfile.TemporaryDirectory(prefix="qrc-tikz-") as build:
        subprocess.run(
            ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
             "-halt-on-error", f"-output-directory={build}", str(tikz_source)],
            check=True, capture_output=True, text=True,
        )
        shutil.copyfile(Path(build) / pdf.name, pdf)
        subprocess.run(
            ["pdftoppm", "-singlefile", "-r", "320", "-png", str(pdf),
             str(png.with_suffix(""))],
            check=True, capture_output=True, text=True,
        )
    source_payload = {
        "historical_input_audit": {
            "path": str(historical_source.relative_to(REPO_ROOT)),
            "sha256": historical_sha256,
            "status": "historical snapshot retained for the prospective N5 seed audit; "
                      "not the source description of the current drawing",
        },
        "input_timeseries": input_timeseries,
        "selected_design": "variant_a_architectures",
        "removed_history_error_panel": input_timeseries["removed_history_error_panel"],
        "canvas_aspect_ratio": 17.9197 / 9.60,
        "canvas_bounds_cm": [0.0, 3.60, 17.9197, 13.2],
        "figure_question": (
            "when can explicit finite context reproduce recurrent input-output behavior"
        ),
        "panels": {
            "architecture": "recurrent and reset-window architectures with unchanged illustrative input and causal access",
            "fading": "upper output frame: fading old-bit influence as time advances at fixed d=40",
            "persistent": "lower output frame: gradual revelation of a persistent old bit as time advances at the same fixed d=40",
        },
        "panel_letters": [],
        "theorem_context": {
            "status": "manuscript interpretation, not a plotted curve or measured property of these toy traces",
            "common_suffix_diameter": "supremal declared-output distance between histories sharing the last d inputs",
            "scalar_information_limit": "D_d/2 is the optimal worst-case absolute scalar error due to missing history",
            "emulation_statement": (
                "a growing exact-expectation reset-QELM family universal on each finite "
                "window emulates uniformly to arbitrary accuracy exactly when D_d tends to zero"
            ),
            "regimes": {
                "fading": {
                    "condition": "D_d tends to zero; r_vis < 1 under the spectral theorem assumptions",
                    "consequence": "arbitrary accuracy as d grows, without a guarantee of exact emulation at a fixed finite d",
                },
                "finite": {
                    "condition": "D_d = 0 for some finite d",
                    "consequence": "output exactly determined by finite context",
                },
                "persistent": {
                    "condition": "D_d >= Delta > 0 for every d; r_vis = 1",
                    "consequence": "scalar worst-case error is at least Delta/2",
                },
            },
            "memory_relevance": (
                "past inputs must create the state difference and present or future "
                "declared outputs must reveal it; full-state forgetting is not required"
            ),
            "scope": (
                "finite-dimensional QRC with input-continuous channels and fixed finite "
                "observable output; the iff statement additionally assumes compact inputs "
                "with freely concatenable suffixes and per-window reset-QELM density"
            ),
            "illustration_boundary": (
                "The displayed time-dependent two-history gaps are not generic D_d/2 curves. "
                "Decay with elapsed time at fixed d does not establish uniform convergence as d grows."
            ),
        },
        "schematic_curves": True,
        "output_comparisons": input_timeseries["output_comparisons"],
        "scope": "Two analytic balanced-bit illustrations with known case-specific baselines, ideal window conditional means, "
                 "fixed d and evolving t; not simulated, fitted or measured QRC/QELM performance.",
        "upstream_dark_sector_summary_sha256": _sha256(dark_summary_path),
        "tikz_source_sha256": _sha256(tikz_source),
        "typography": "native 11 pt labels and one shared 10 pt legend at 7.055 inch width; no panel letters",
        "history_access": "the interior point at t=180 supplies the current scalar; "
                          "older inputs act through the carried-state loop; the exact causal "
                          "window is 141--180 and grey future samples 181--200 are display-only",
        "past_input_cue": "blue!16 behind the older-input interval, ending where the green!24 window starts; "
                          "a 0.4-pt solid green!65!ink boundary marks index 140.5, distinct from the dashed current-time guide; "
                          "blue brace labeled past carried in / QRC state rho_(t-1); "
                          "not a second input wire, history replay, or claim of lossless history storage",
        "lane_spacing": "the 4.15 cm square input and all architecture coordinates retain their positions and native labels; "
                        "both dynamics groups retain their 0.35 cm rightward offset and observable banks their 0.15 cm offset; "
                        "the orthogonal scalar connector enters the QRC top center vertically downward; "
                        "the reset lane is 0.30 cm lower to separate the model rows; "
                        "a coral causal-window brace feeds the reset encoder from below without crossing quantum wires; "
                        "two 4.15-by-3.2 cm output frames occupy y=8.58..11.78 and y=4.99..8.19 with matching axes; "
                        "case labels are omitted; a single shared legend overlaps their facing borders; "
                        "the removed lower history-error panel permits a 17.9197-by-9.60 cm canvas",
        "architecture_motifs": "three-wire general QRC channel block and explicit reset encoding plus fixed gates; "
                               "matching yellow observable banks with softly rounded cards and connectors; "
                               "faint downstream lanes denote omitted intermediate observables, not extra quantum wires; "
                               "compact rounded unlabeled readout triangles; both outputs feed the paired memory-case comparisons",
        "readout_scope": "fixed target readout (identity for the declared observable vector); "
                         "trained reset-QELM emulator readout; operations remain defined by the caption and model section "
                         "while endpoints are labeled y_t and its hatted counterpart, with no trained-readout text",
        "observable_notation": "O_1,...,O_m label target observables and P_1,...,P_B label QELM features; "
                               "no identical-operator or equal-feature-count claim",
        "annotation_scope": "no standalone zoom or schematic tags; one unchanged input amplitude scale; "
                            "no Fading memory or Persistent memory titles and no external expiry/explanation labels; "
                            "green!24 before d and blue!16 after d inside both output frames encode the old bit's window access; "
                            "a single one-row reference-style key overlaps the facing borders between both frames "
                            "and identifies QRC and QELM, with no independent third truth trajectory; "
                            "time tick labels appear only on the lower frame; input-step and time axis titles are omitted; "
                            "no panel letters, epsilon or D_d/2 appear in the artwork; "
                            "caption identifies the ideal toy reference, illustrative status and future exclusion",
    }
    source.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def _validated_history_frontier(source_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_path)
    keys = ["teacher_n_qubits", "teacher_gamma", "window"]
    required = {
        *keys,
        "median",
        "minimum",
        "maximum",
        "n_roots",
        "suffix_live_n",
        "suffix_dead_n",
        "provenance_config_sha256",
        "provenance_decision",
    }
    if missing := required - set(frame):
        raise ValueError(f"history-frontier source is missing columns: {sorted(missing)}")
    if len(frame) != 48 or frame.duplicated(keys).any():
        raise ValueError("history-frontier source must contain 48 unique tested cells")
    if set(frame["teacher_n_qubits"].astype(int)) != {2, 3}:
        raise ValueError("history-frontier source has an unexpected teacher grid")
    if set(frame["teacher_gamma"].astype(float)) != {0.0, 0.2, 0.5, 1.0, 2.0, 4.0}:
        raise ValueError("history-frontier source has an unexpected loss grid")
    if set(frame["window"].astype(int)) != {1, 2, 4, 6}:
        raise ValueError("history-frontier source has an unexpected window grid")
    values = frame[["minimum", "median", "maximum"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("history-frontier NRMSE values must be finite and positive")
    if np.any(values[:, 0] > values[:, 1]) or np.any(values[:, 1] > values[:, 2]):
        raise ValueError("history-frontier minimum/median/maximum ordering is invalid")
    if not (frame["n_roots"].astype(int) == 12).all():
        raise ValueError("history-frontier source must use 12 root blocks per cell")
    if not (frame["suffix_live_n"].astype(int) == 12).all() or not (
        frame["suffix_dead_n"].astype(int) == 0
    ).all():
        raise ValueError("history-frontier source must retain all 12 live roots")
    return frame.sort_values(keys).reset_index(drop=True)


def _validated_dense_phase_source(
    source_path: Path,
    dense_suffix_source_path: Path = DEFAULT_DENSE_SUFFIX_SOURCE,
) -> pd.DataFrame:
    frame = pd.read_csv(source_path)
    keys = ["teacher_n_qubits", "teacher_gamma", "window"]
    required = {
        *keys,
        "median_history_nrmse",
        "minimum_history_nrmse",
        "maximum_history_nrmse",
        "history_n_roots",
        "state",
        "minimum_feature_budget",
        "any_root_disagreement",
        "minimum_agreement_n",
        "emulated_n_B18",
        "emulated_n_B36",
        "emulated_n_B72",
        "run_manifest_sha256",
        "root_set_sha256",
        "config_sha256",
        "suffix_decomposition_sha256",
        "freeze_record_sha256",
    }
    if missing := required - set(frame):
        raise ValueError(f"dense phase source is missing columns: {sorted(missing)}")
    if len(frame) != 132 or frame.duplicated(keys).any():
        raise ValueError("dense phase source must contain 132 unique cells")
    if set(frame["teacher_n_qubits"].astype(int)) != {2, 3}:
        raise ValueError("dense phase source has an unexpected teacher grid")
    if set(frame["teacher_gamma"].astype(float)) != {
        0.0,
        0.2,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
    }:
        raise ValueError("dense phase source has an unexpected loss grid")
    if set(frame["window"].astype(int)) != {1, 2, 3, 4, 5, 6}:
        raise ValueError("dense phase source has an unexpected window grid")
    values = frame[
        ["minimum_history_nrmse", "median_history_nrmse", "maximum_history_nrmse"]
    ].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("dense history NRMSE values must be finite and positive")
    if np.any(values[:, 0] > values[:, 1]) or np.any(values[:, 1] > values[:, 2]):
        raise ValueError("dense history minimum/median/maximum ordering is invalid")
    if not frame["history_n_roots"].astype(int).eq(12).all():
        raise ValueError("dense phase source must use 12 roots in every cell")
    states = {"HISTORY_MISSING", "FEASIBLE", "MAP_NOT_REPRESENTED", "UNRESOLVED"}
    if not set(frame["state"].astype(str)).issubset(states):
        raise ValueError("dense phase source contains an unsupported state")
    emulated_columns = {18: "emulated_n_B18", 36: "emulated_n_B36", 72: "emulated_n_B72"}
    for column in emulated_columns.values():
        counts = frame[column].astype(int)
        if ((counts < 0) | (counts > 12)).any():
            raise ValueError(f"dense phase source has invalid {column}")
    for row in frame.itertuples(index=False):
        successes = [
            budget
            for budget, column in emulated_columns.items()
            if int(getattr(row, column)) >= 10
        ]
        if successes:
            if row.state != "FEASIBLE" or int(row.minimum_feature_budget) != min(successes):
                raise ValueError("dense robust feature badge is inconsistent")
        elif row.state == "FEASIBLE":
            raise ValueError("dense feasible cell lacks 10-of-12 root support")
        if row.state == "HISTORY_MISSING" and float(row.median_history_nrmse) <= 0.10:
            raise ValueError("dense history-missing state is below the frozen tolerance")
    if not np.equal(
        frame["any_root_disagreement"].astype(str).str.lower().map({"true": 1, "false": 0}),
        frame["minimum_agreement_n"].astype(int).lt(12).astype(int),
    ).all():
        raise ValueError("dense root-disagreement flag is inconsistent")
    for column in (
        "run_manifest_sha256",
        "root_set_sha256",
        "config_sha256",
        "suffix_decomposition_sha256",
        "freeze_record_sha256",
    ):
        values = frame[column].astype(str)
        if values.nunique(dropna=False) != 1 or not values.str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"dense phase source has invalid {column}")

    expected_suffix_hash = str(frame["suffix_decomposition_sha256"].iloc[0])
    if _sha256(dense_suffix_source_path) != expected_suffix_hash:
        raise ValueError("dense suffix source does not match its registered SHA-256")
    suffix = pd.read_csv(dense_suffix_source_path)
    suffix_required = {
        "root_id",
        *keys,
        "model",
        "history_nrmse",
    }
    if missing := suffix_required - set(suffix):
        raise ValueError(f"dense suffix source is missing columns: {sorted(missing)}")
    history = suffix[suffix["model"].eq("rr_same_identity")].copy()
    history_keys = ["root_id", *keys]
    if len(history) != 1_584 or history.duplicated(history_keys).any():
        raise ValueError(
            "dense suffix history slice must contain 12 roots x 132 unique cells"
        )
    if set(history["root_id"].astype(str)) != {
        f"dense{index:02d}" for index in range(1, 13)
    }:
        raise ValueError("dense suffix source has an unexpected root set")
    if set(history["teacher_n_qubits"].astype(int)) != {2, 3} or set(
        history["teacher_gamma"].astype(float)
    ) != set(frame["teacher_gamma"].astype(float)) or set(
        history["window"].astype(int)
    ) != {1, 2, 3, 4, 5, 6}:
        raise ValueError("dense suffix source has an unexpected tested grid")
    history_values = history["history_nrmse"].to_numpy(dtype=float)
    if not np.all(np.isfinite(history_values)) or np.any(history_values <= 0.0):
        raise ValueError("dense suffix history NRMSE values must be finite and positive")

    history_counts = (
        history.groupby(keys, as_index=False)
        .agg(
            history_sufficient_n=(
                "history_nrmse",
                lambda values: int((values <= 0.10).sum()),
            ),
            history_root_count=("root_id", "nunique"),
            recomputed_median_history_nrmse=("history_nrmse", "median"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    if len(history_counts) != 132 or not history_counts["history_root_count"].eq(12).all():
        raise ValueError("dense suffix history counts are incomplete")
    frame = frame.merge(history_counts, on=keys, how="left", validate="one_to_one")
    if frame[["history_sufficient_n", "history_root_count"]].isna().any().any():
        raise ValueError("dense suffix history counts do not cover every phase cell")
    if not np.allclose(
        frame["median_history_nrmse"].to_numpy(dtype=float),
        frame["recomputed_median_history_nrmse"].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-13,
    ):
        raise ValueError("dense suffix history medians disagree with the phase source")
    frame = frame.drop(columns="recomputed_median_history_nrmse")
    frame["history_sufficient_n"] = frame["history_sufficient_n"].astype(int)
    frame["history_root_count"] = frame["history_root_count"].astype(int)

    def display_outcome(row: pd.Series) -> str:
        if int(row["history_sufficient_n"]) < 10:
            return "HISTORY_NOT_ROBUST"
        if str(row["state"]) == "FEASIBLE":
            return f"QELM_{int(row['minimum_feature_budget'])}"
        return "NO_TESTED_QELM_ROBUST"

    frame["display_outcome"] = frame.apply(display_outcome, axis=1)
    feasible = frame["state"].eq("FEASIBLE")
    if not frame.loc[feasible, "history_sufficient_n"].ge(10).all():
        raise ValueError("a robust QELM success lacks robust history sufficiency")
    allowed_display = {
        "HISTORY_NOT_ROBUST",
        "QELM_18",
        "QELM_36",
        "QELM_72",
        "NO_TESTED_QELM_ROBUST",
    }
    if not set(frame["display_outcome"]).issubset(allowed_display):
        raise ValueError("dense phase source produces an unsupported display outcome")
    return frame.sort_values(keys).reset_index(drop=True)


def _validated_dense_n4_phase_source(
    source_path: Path,
    suffix_source_path: Path = DEFAULT_DENSE_N4_SUFFIX_SOURCE,
) -> pd.DataFrame:
    """Validate the separately frozen prospective four-qubit follow-up."""

    frame = pd.read_csv(source_path)
    keys = ["teacher_n_qubits", "teacher_gamma", "window"]
    required = {
        *keys,
        "median_history_nrmse",
        "minimum_history_nrmse",
        "maximum_history_nrmse",
        "history_n_roots",
        "history_sufficient_n",
        "state",
        "minimum_feature_budget",
        "any_root_disagreement",
        "minimum_agreement_n",
        "emulated_n_B18",
        "emulated_n_B36",
        "emulated_n_B72",
        "evidence_cohort",
        "run_manifest_sha256",
        "root_set_sha256",
        "config_sha256",
        "suffix_decomposition_sha256",
        "freeze_record_sha256",
    }
    if missing := required - set(frame):
        raise ValueError(f"dense N4 phase source is missing columns: {sorted(missing)}")
    if len(frame) != 66 or frame.duplicated(keys).any():
        raise ValueError("dense N4 phase source must contain 66 unique cells")
    if set(frame["teacher_n_qubits"].astype(int)) != {4}:
        raise ValueError("dense N4 phase source has an unexpected teacher grid")
    expected_gammas = {
        0.0,
        0.2,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
    }
    if set(frame["teacher_gamma"].astype(float)) != expected_gammas:
        raise ValueError("dense N4 phase source has an unexpected loss grid")
    if set(frame["window"].astype(int)) != {1, 2, 3, 4, 5, 6}:
        raise ValueError("dense N4 phase source has an unexpected window grid")
    values = frame[
        ["minimum_history_nrmse", "median_history_nrmse", "maximum_history_nrmse"]
    ].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("dense N4 history NRMSE values must be finite and nonnegative")
    if np.any(values[:, 0] > values[:, 1]) or np.any(values[:, 1] > values[:, 2]):
        raise ValueError("dense N4 history minimum/median/maximum ordering is invalid")
    if not frame["history_n_roots"].astype(int).eq(12).all():
        raise ValueError("dense N4 phase source must use 12 roots in every cell")
    if not frame["history_sufficient_n"].astype(int).between(0, 12).all():
        raise ValueError("dense N4 phase source has invalid history support counts")
    if not frame["evidence_cohort"].eq("prospective_dense_n4_followup").all():
        raise ValueError("dense N4 evidence-cohort label is invalid")
    states = {"HISTORY_MISSING", "FEASIBLE", "MAP_NOT_REPRESENTED", "UNRESOLVED"}
    if not set(frame["state"].astype(str)).issubset(states):
        raise ValueError("dense N4 phase source contains an unsupported state")
    emulated_columns = {18: "emulated_n_B18", 36: "emulated_n_B36", 72: "emulated_n_B72"}
    for column in emulated_columns.values():
        counts = frame[column].astype(int)
        if ((counts < 0) | (counts > 12)).any():
            raise ValueError(f"dense N4 phase source has invalid {column}")
    for row in frame.itertuples(index=False):
        successes = [
            budget
            for budget, column in emulated_columns.items()
            if int(getattr(row, column)) >= 10
        ]
        if successes:
            if row.state != "FEASIBLE" or int(row.minimum_feature_budget) != min(successes):
                raise ValueError("dense N4 robust feature badge is inconsistent")
        elif row.state == "FEASIBLE":
            raise ValueError("dense N4 feasible cell lacks 10-of-12 root support")
        if row.state == "HISTORY_MISSING" and int(row.history_sufficient_n) >= 10:
            raise ValueError("dense N4 history-missing state has robust history support")
    disagreement = frame["any_root_disagreement"].astype(str).str.lower().map(
        {"true": 1, "false": 0}
    )
    if disagreement.isna().any() or not np.equal(
        disagreement,
        frame["minimum_agreement_n"].astype(int).lt(12).astype(int),
    ).all():
        raise ValueError("dense N4 root-disagreement flag is inconsistent")
    hash_columns = (
        "run_manifest_sha256",
        "root_set_sha256",
        "config_sha256",
        "suffix_decomposition_sha256",
        "freeze_record_sha256",
    )
    for column in hash_columns:
        hashes = frame[column].astype(str)
        if hashes.nunique(dropna=False) != 1 or not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"dense N4 phase source has invalid {column}")

    expected_suffix_hash = str(frame["suffix_decomposition_sha256"].iloc[0])
    if _sha256(suffix_source_path) != expected_suffix_hash:
        raise ValueError("dense N4 suffix source does not match its registered SHA-256")
    suffix = pd.read_csv(suffix_source_path)
    suffix_required = {"root_id", *keys, "model", "history_nrmse"}
    if missing := suffix_required - set(suffix):
        raise ValueError(f"dense N4 suffix source is missing columns: {sorted(missing)}")
    history = suffix[suffix["model"].eq("rr_same_identity")].copy()
    history_keys = ["root_id", *keys]
    if len(history) != 792 or history.duplicated(history_keys).any():
        raise ValueError("dense N4 suffix history slice must contain 12 roots x 66 cells")
    if set(history["root_id"].astype(str)) != {
        f"dense_n4_{index:02d}" for index in range(1, 13)
    }:
        raise ValueError("dense N4 suffix source has an unexpected root set")
    if set(history["teacher_n_qubits"].astype(int)) != {4} or set(
        history["teacher_gamma"].astype(float)
    ) != expected_gammas or set(history["window"].astype(int)) != {1, 2, 3, 4, 5, 6}:
        raise ValueError("dense N4 suffix source has an unexpected tested grid")
    history_values = history["history_nrmse"].to_numpy(dtype=float)
    if not np.all(np.isfinite(history_values)) or np.any(history_values < 0.0):
        raise ValueError("dense N4 suffix history NRMSE values must be finite and nonnegative")
    recomputed = (
        history.groupby(keys, as_index=False)
        .agg(
            recomputed_history_sufficient_n=(
                "history_nrmse",
                lambda observations: int((observations <= 0.10).sum()),
            ),
            history_root_count=("root_id", "nunique"),
            recomputed_median_history_nrmse=("history_nrmse", "median"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    if len(recomputed) != 66 or not recomputed["history_root_count"].eq(12).all():
        raise ValueError("dense N4 suffix history counts are incomplete")
    frame = frame.merge(recomputed, on=keys, how="left", validate="one_to_one")
    if frame[["recomputed_history_sufficient_n", "history_root_count"]].isna().any().any():
        raise ValueError("dense N4 suffix history counts do not cover every phase cell")
    if not frame["history_sufficient_n"].astype(int).eq(
        frame["recomputed_history_sufficient_n"].astype(int)
    ).all():
        raise ValueError("dense N4 history support counts disagree with the raw suffix table")
    if not np.allclose(
        frame["median_history_nrmse"].to_numpy(dtype=float),
        frame["recomputed_median_history_nrmse"].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-13,
    ):
        raise ValueError("dense N4 suffix history medians disagree with the phase source")
    frame = frame.drop(
        columns=["recomputed_history_sufficient_n", "recomputed_median_history_nrmse"]
    )

    def display_outcome(row: pd.Series) -> str:
        if int(row["history_sufficient_n"]) < 10:
            return "HISTORY_NOT_ROBUST"
        if str(row["state"]) == "FEASIBLE":
            return f"QELM_{int(row['minimum_feature_budget'])}"
        return "NO_TESTED_QELM_ROBUST"

    frame["display_outcome"] = frame.apply(display_outcome, axis=1)
    return frame.sort_values(keys).reset_index(drop=True)


def _validated_dense_merged_source(
    dense_phase_source_path: Path,
    dense_suffix_source_path: Path,
    dense_n4_phase_source_path: Path,
    dense_n4_suffix_source_path: Path,
) -> pd.DataFrame:
    earlier = _validated_dense_phase_source(
        dense_phase_source_path,
        dense_suffix_source_path,
    ).assign(evidence_cohort="frozen_dense_n2_n3")
    followup = _validated_dense_n4_phase_source(
        dense_n4_phase_source_path,
        dense_n4_suffix_source_path,
    )
    for column in ("run_manifest_sha256", "root_set_sha256", "config_sha256"):
        if str(earlier[column].iloc[0]) == str(followup[column].iloc[0]):
            raise ValueError(f"dense cohorts unexpectedly share {column}")
    merged = pd.concat([earlier, followup], ignore_index=True, sort=False)
    keys = ["teacher_n_qubits", "teacher_gamma", "window"]
    if len(merged) != 198 or merged.duplicated(keys).any():
        raise ValueError("merged dense display source must contain 198 unique cells")
    if merged["evidence_cohort"].value_counts().to_dict() != {
        "frozen_dense_n2_n3": 132,
        "prospective_dense_n4_followup": 66,
    }:
        raise ValueError("merged dense evidence cohorts are incomplete")
    return merged.sort_values(keys).reset_index(drop=True)


def _draw_exact_threshold_boundary(ax: Any, below: np.ndarray) -> None:
    """Draw cell-edge segments only where tested neighbors straddle tolerance."""

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    rows, columns = below.shape
    for y in range(rows):
        for x in range(columns - 1):
            if bool(below[y, x]) != bool(below[y, x + 1]):
                segments.append(((x + 0.5, y - 0.5), (x + 0.5, y + 0.5)))
    for y in range(rows - 1):
        for x in range(columns):
            if bool(below[y, x]) != bool(below[y + 1, x]):
                segments.append(((x - 0.5, y + 0.5), (x + 0.5, y + 0.5)))
    for start, end in segments:
        ax.plot(*zip(start, end), color=INK, linewidth=1.65, solid_capstyle="butt", zorder=3.0)
        ax.plot(*zip(start, end), color=WHITE, linewidth=0.92, solid_capstyle="butt", zorder=3.1)


def render_figure_2(
    dense_phase_source_path: Path,
    dense_suffix_source_path: Path,
    dense_n4_phase_source_path: Path,
    dense_n4_suffix_source_path: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Combine history and robust emulation status at the same measured cells.

    The line color encodes context, never the feature budget. Marker shape
    carries the unchanged root-level classification, including cells whose
    median crosses the target without passing the 10-of-12 history rule.
    """
    table = _validated_dense_merged_source(
        dense_phase_source_path,
        dense_suffix_source_path,
        dense_n4_phase_source_path,
        dense_n4_suffix_source_path,
    )
    _base_style()
    fig = plt.figure(figsize=FIGURE_SIZES["fig2"], facecolor=WHITE)
    windows = (1, 2, 3, 4, 5, 6)
    # Viridis colors extracted from the supplied reference's vector artwork.
    window_colors = ("#7AD151", "#44BF70", "#1FA088", "#2A788E", "#414487", "#482475")
    outcome_styles = {
        "HISTORY_NOT_ROBUST": ("o", 2.6, False),
        "NO_TESTED_QELM_ROBUST": ("x", 4.5, True),
        "QELM_18": ("o", 4.3, True),
        "QELM_36": ("s", 4.3, True),
        "QELM_72": ("D", 4.3, True),
    }
    width = 0.27
    height = width * fig.get_figwidth() / fig.get_figheight()
    axes = [fig.add_axes([left, 0.225, width, height]) for left in (0.10, 0.41, 0.72)]
    for panel, (ax, n_qubits) in enumerate(zip(axes, (2, 3, 4))):
        ax.set_box_aspect(1)
        family = table[table["teacher_n_qubits"].eq(n_qubits)]
        for window, color in zip(windows, window_colors):
            curve = family[family["window"].eq(window)].sort_values("teacher_gamma")
            ax.plot(curve["teacher_gamma"], curve["median_history_nrmse"],
                    color=color, linewidth=1.0, zorder=2)
            for outcome, (marker, size, filled) in outcome_styles.items():
                selected = curve[curve["display_outcome"].eq(outcome)]
                ax.plot(
                    selected["teacher_gamma"], selected["median_history_nrmse"],
                    linestyle="none", marker=marker, markersize=size,
                    markerfacecolor=color if filled else WHITE,
                    markeredgecolor=color, markeredgewidth=0.85 if marker == "x" else 0.65,
                    zorder=4,
                )
            endpoint = curve.iloc[-1]
            ax.text(4.12, float(endpoint["median_history_nrmse"]), f"$d={window}$",
                    color=INK, fontsize=8.0, va="center", ha="left")
        ax.set_yscale("log")
        ax.set_ylim(7e-4, 3.0)
        ax.set_xlim(-0.08, 4.90)
        ax.set_xticks((0, 1, 2, 3, 4))
        ax.set_yticks((0.001, 0.01, 0.10, 1.0), ("0.1%", "1%", "10%", "100%"))
        ax.axhline(0.10, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.75, zorder=1)
        if panel == 0:
            ax.set_ylabel("old-history error", labelpad=4)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel(r"loss rate $\gamma_T$", labelpad=3)
        ax.text(0.035, 0.97, chr(97 + panel), transform=ax.transAxes,
                va="top", fontweight="bold", fontsize=STYLE["axes_titlesize"])
        ax.text(0.965, 0.97, f"{n_qubits}-qubit QRC", transform=ax.transAxes,
                va="top", ha="right", fontsize=9.0)
        _style_quant_axis(ax)

    # Each explanatory key occupies unused low-error space. Context lengths
    # are labeled on the curves, so a second six-entry legend is unnecessary.
    marker_keys = [
        [
            Line2D([], [], linestyle="none", marker="o", markerfacecolor=WHITE,
                   markeredgecolor=INK, markersize=3.0, label="history unmet"),
            Line2D([], [], linestyle="none", marker="x", color=INK,
                   markersize=4.5, label="QELM unmet"),
        ],
        [
            Line2D([], [], linestyle="none", marker=marker, color=INK,
                   markersize=4.3, label=f"{budget} features")
            for budget, marker in ((18, "o"), (36, "s"), (72, "D"))
        ],
        [
            Line2D([], [], color=MUTED, linestyle=(0, (4, 3)), linewidth=0.75,
                   label="10% target"),
        ],
    ]
    for ax, handles in zip(axes, marker_keys):
        legend = ax.legend(handles=handles, loc="lower left", fontsize=8.0,
                           frameon=True, fancybox=True, framealpha=0.97,
                           handlelength=1.0, handletextpad=0.4, borderpad=0.35,
                           labelspacing=0.3, borderaxespad=0.5)
        _style_legend(legend)
    pdf = output_dir / "fig02_information_frontier.pdf"
    png = output_dir / "fig02_information_frontier.png"
    source = output_dir / "fig02_information_frontier_source.csv"
    table.to_csv(source, index=False)
    fig.savefig(pdf, metadata=_metadata("History curves with robust QELM outcomes"))
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


def _comparison_axes(fig: plt.Figure, count: int) -> list[plt.Axes]:
    """Match physical frames and gutters for the two main evidence figures."""
    if count not in (2, 3):
        raise ValueError("The publication comparison row contains two or three panels")
    side = COMPARISON_LAYOUT["panel_side"]
    gap = COMPARISON_LAYOUT["panel_gap" if count == 3 else "two_panel_gap"]
    stride = side + gap
    row_center = COMPARISON_LAYOUT["three_panel_left"] + (3 * side + 2 * COMPARISON_LAYOUT["panel_gap"]) / 2
    left = row_center - (count * side + (count - 1) * gap) / 2
    axes = [fig.add_axes([
        (left + index * stride) / fig.get_figwidth(),
        COMPARISON_LAYOUT["bottom"] / fig.get_figheight(),
        side / fig.get_figwidth(),
        side / fig.get_figheight(),
    ]) for index in range(count)]
    for ax in axes:
        ax.set_box_aspect(1)
    return axes


def _resource_diagnosis_axes(fig: plt.Figure) -> list[plt.Axes]:
    """Keep three matched, physically square resource panels in one row."""
    return _comparison_axes(fig, 3)


def _feature_density_source(path: Path) -> pd.DataFrame:
    """Only the complete, separately frozen follow-up enters the new panel."""
    from run_feature_density import coverage, verify
    frozen = verify()
    manifest = json.loads(path.with_name("manifest.json").read_text())
    if (manifest.get("status") != "COMPLETED" or manifest.get("smoke") is not False
            or manifest.get("freeze") != frozen or manifest.get("material") != frozen["material"]
            or manifest.get("metrics_sha256") != _sha256(path)):
        raise ValueError("feature-density source provenance mismatch")
    frame = pd.read_csv(path, float_precision="round_trip")
    coverage(frame)
    if not np.allclose(frame["total_nrmse"], np.sqrt(frame["total_mse"]/frame["target_variance"]), atol=1e-12):
        raise ValueError("feature-density normalization mismatch")
    return frame[frame["model"].eq("ptd_parallel")].assign(
        cohort="feature_density_followup", source_sha256=_sha256(path))


def _feature_density_with_bridge(path: Path) -> pd.DataFrame:
    """Add only new partial banks; replicated 18/36 controls are not duplicates."""
    from feature_bridge_release import RESULTS, load_completed

    original = _feature_density_source(path)
    bridge, manifest = load_completed()
    new = bridge[bridge["feature_budget"].isin((24, 30))].assign(
        cohort="feature_density_followup", source_sha256=manifest["metrics_sha256"])
    if len(new) != 72 or _sha256(RESULTS / "metrics.csv") != manifest["metrics_sha256"]:
        raise ValueError("intermediate feature coverage or identity mismatch")
    combined = pd.concat([original.assign(experiment_stage="original_six_budget_followup"), new],
                         ignore_index=True, sort=False)
    if len(combined) != 288 or combined.duplicated(["root_id", "teacher_n_qubits", "feature_budget"]).any():
        raise ValueError("duplicate or incomplete combined feature source")
    return combined


def _resource_history_with_n4(scaling: pd.DataFrame) -> pd.DataFrame:
    _validated_dense_n4_phase_source(DEFAULT_DENSE_N4_PHASE_SOURCE, DEFAULT_DENSE_N4_SUFFIX_SOURCE)
    fourth = pd.read_csv(DEFAULT_DENSE_N4_SUFFIX_SOURCE)
    fourth = fourth[fourth["model"].eq("rr_same_identity")
                    & fourth["teacher_gamma"].isin((.2,.5,1.,2.,4.))
                    & fourth["window"].isin((1,2,4,6))].copy()
    if len(fourth) != 240 or fourth.duplicated(["root_id","teacher_gamma","window"]).any():
        raise ValueError("four-qubit scaling slice is incomplete")
    fourth["scaling_x"] = fourth["teacher_gamma"]*.5*fourth["window"]
    fourth["cohort"] = "independent_dense_n4"
    fourth["source_sha256"] = _sha256(DEFAULT_DENSE_N4_SUFFIX_SOURCE)
    original = scaling.assign(cohort="canonical_confirmatory", source_sha256=_sha256(DEFAULT_SCALING_SOURCE))
    columns=["root_id","teacher_n_qubits","teacher_gamma","window","scaling_x","history_nrmse","cohort","source_sha256"]
    return pd.concat([original[columns],fourth[columns]],ignore_index=True)


def _finish_comparison_axis(ax: plt.Axes, letter: str) -> None:
    """Use the same frame, letter anchor and label spacing in Figures 2/3."""
    _style_quant_axis(ax)
    ax.text(0.045, 0.965, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=STYLE["axes_titlesize"], fontweight="bold")
    ax.xaxis.labelpad = 4.0
    ax.yaxis.labelpad = 3.0


def render_figure_3(
    scaling_source_path: Path,
    suffix_source_path: Path,
    shot_source_path: Path,
    output_dir: Path,
    feature_density_source_path: Path = DEFAULT_FEATURE_DENSITY_SOURCE,
) -> tuple[Path, ...]:
    scaling = _resource_history_with_n4(validated_scaling_source(scaling_source_path))
    _validated_suffix_features(suffix_source_path)  # retain the historical-source audit
    features = _feature_density_with_bridge(feature_density_source_path)
    shots = _validated_shot_effects(shot_source_path)
    _base_style()
    fig = plt.figure(figsize=FIGURE_SIZES["fig3"], facecolor=WHITE)
    axes = _resource_diagnosis_axes(fig)
    family_colors = colors.QRC_SIZES
    family_markers = {2: "o", 3: "D", 4: "s"}
    family_handles = [
        Line2D([], [], color=family_colors[n], marker=family_markers[n],
               markerfacecolor=WHITE if n == 2 else family_colors[n],
               markersize=3.7, markeredgewidth=0.6,
               linewidth=STYLE["primary_linewidth"], label=f"{n}-qubit QRC")
        for n in (2, 3, 4)
    ]

    ax = axes[0]
    root_scaling = (
        scaling.groupby(["teacher_n_qubits", "root_id", "scaling_x", "cohort", "source_sha256"], as_index=False)["history_nrmse"]
        .median()
    )
    for n_qubits in (2, 3, 4):
        family = root_scaling[root_scaling["teacher_n_qubits"].eq(n_qubits)]
        summary = family.groupby("scaling_x")["history_nrmse"].agg(
            median="median", q10=lambda values: values.quantile(0.10), q90=lambda values: values.quantile(0.90)
        ).reset_index()
        ax.errorbar(
            summary["scaling_x"],
            summary["median"],
            yerr=np.vstack([summary["median"] - summary["q10"],
                            summary["q90"] - summary["median"]]),
            capsize=1.5, capthick=0.7, elinewidth=0.7,
            color=family_colors[n_qubits],
            marker=family_markers[n_qubits],
            markersize=3.7,
            markerfacecolor=WHITE if n_qubits == 2 else family_colors[n_qubits],
            markeredgewidth=0.6,
            linewidth=STYLE["primary_linewidth"],
            label=f"{n_qubits}-qubit QRC",
            zorder=3,
        )
    ax.axhline(0.10, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.85)
    ax.text(
        11.8,
        0.070,
        "10% target",
        ha="right",
        va="top",
        fontsize=9.0,
        color=MUTED,
    )
    ax.set_yscale("log")
    ax.set_xlim(-0.30, 12.50)
    ax.set_ylim(5e-4, 3.2)  # small headroom keeps the legend above the 10% line
    ax.set_yticks((0.001, 0.01, 0.1, 1), ("0.1%", "1%", "10%", "100%"))
    ax.set_xticks((0, 3, 6, 9, 12))
    ax.set_xlabel("scaled context $\\gamma_T\\,\\Delta t\\,d$")
    ax.set_ylabel("old-history error")
    legend = ax.legend(
        handles=family_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.01),
        **COMPARISON_LEGEND,
    )
    _style_legend(legend)

    ax = axes[1]
    for n_qubits in (2, 3, 4):
        # Keep all eight measured budgets in the source; display the transition.
        family = features[features["teacher_n_qubits"].eq(n_qubits)
                          & features["feature_budget"].le(72)]
        summary = family.groupby("feature_budget")["total_nrmse"].agg(
            median="median", q10=lambda values: values.quantile(0.10),
            q90=lambda values: values.quantile(0.90),
        ).reset_index()
        ax.errorbar(
            summary["feature_budget"],
            summary["median"],
            yerr=np.vstack([summary["median"] - summary["q10"],
                            summary["q90"] - summary["median"]]),
            capsize=1.5, capthick=0.7, elinewidth=0.7,
            color=family_colors[n_qubits],
            marker=family_markers[n_qubits],
            markersize=3.7,
            markerfacecolor=WHITE if n_qubits == 2 else family_colors[n_qubits],
            markeredgewidth=0.6,
            linewidth=STYLE["primary_linewidth"],
            label=f"{n_qubits}-qubit QRC",
        )
    ax.axhline(0.10, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.85)
    ax.text(
        70.0,
        0.104,
        "10% target",
        ha="right",
        va="bottom",
        fontsize=9.0,
        color=MUTED,
    )
    ax.set_ylim(0.035, 0.325)
    ax.set_xlim(14, 76)
    ax.set_xticks((18, 24, 30, 36, 54, 72))
    ax.tick_params(axis="x", labelsize=8.0)
    ax.set_yticks((0.05, 0.10, 0.15, 0.20, 0.25), ("5%", "10%", "15%", "20%", "25%"))
    ax.set_xlabel("measured features $B$")
    ax.set_ylabel("QELM emulation error")
    legend = ax.legend(
        handles=family_handles,
        loc="upper right",
        **COMPARISON_LEGEND,
    )
    _style_legend(legend)

    ax = axes[2]
    shot_styles = {
        "test_nrmse_B18": (PALETTE["viridis_18"], "o", "-", "18 features"),
        "test_nrmse_B36": (PALETTE["viridis_36"], "s", "-", "36 features"),
        "test_nrmse_B72": (PALETTE["viridis_72"], "D", "-", "72 features"),
        "selected_policy_test_nrmse": (INK, "*", "--", "validation-selected"),
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
            markersize=5.5 if marker == "*" else 3.7,
            markeredgewidth=0.6,
            linewidth=STYLE["primary_linewidth"],
            linestyle=linestyle,
            label=label,
            zorder=5 if column == "selected_policy_test_nrmse" else 3,
        )
        if column == "selected_policy_test_nrmse":
            ax.errorbar(
                summary["total_shots"],
                summary["median"],
                yerr=np.vstack([summary["median"] - summary["q10"],
                                summary["q90"] - summary["median"]]),
                fmt="none", color=INK, capsize=1.5, capthick=0.7,
                elinewidth=0.7, zorder=4,
            )
    ax.set_xscale("log", base=4)
    ax.set_xticks((300, 1200, 4800, 19200), ("300", "1.2k", "4.8k", "19.2k"))
    ax.set_ylim(0.055, 0.45)  # clear space for the legend and below-target label
    ax.set_yticks((0.10, 0.20, 0.30), ("10%", "20%", "30%"))
    ax.set_xlabel("shots per input $Q$")
    ax.set_ylabel("QELM test error")
    ax.axhline(0.10, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.85)
    ax.text(
        300,
        0.097,
        "10% target",
        ha="left",
        va="top",
        fontsize=9.0,
        color=MUTED,
    )
    legend = ax.legend(
        loc="upper right",
        **COMPARISON_LEGEND,
    )
    _style_legend(legend)

    for index, ax in enumerate(axes):
        _finish_comparison_axis(ax, "abc"[index])
    pdf = output_dir / "fig03_resource_diagnosis.pdf"
    png = output_dir / "fig03_resource_diagnosis.png"
    source = output_dir / "fig03_resource_diagnosis_source.csv"

    scaling_export = root_scaling.rename(columns={"history_nrmse": "value"}).assign(panel="history", metric="history_nrmse")
    feature_export = features.rename(columns={"total_nrmse": "value"}).assign(panel="features", metric="total_nrmse")
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
    dense_phase_source_path: Path = DEFAULT_DENSE_PHASE_SOURCE,
    dense_suffix_source_path: Path = DEFAULT_DENSE_SUFFIX_SOURCE,
    dense_n4_phase_source_path: Path = DEFAULT_DENSE_N4_PHASE_SOURCE,
    dense_n4_suffix_source_path: Path = DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    scaling_source_path: Path = DEFAULT_SCALING_SOURCE,
    suffix_source_path: Path = DEFAULT_SUFFIX_SOURCE,
    shot_source_path: Path = DEFAULT_SHOT_SOURCE,
    feature_density_source_path: Path = DEFAULT_FEATURE_DENSITY_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        *render_figure_1(dark_summary_path, output_dir),
        *render_figure_2(
            dense_phase_source_path,
            dense_suffix_source_path,
            dense_n4_phase_source_path,
            dense_n4_suffix_source_path,
            output_dir,
        ),
        *render_figure_3(scaling_source_path, suffix_source_path, shot_source_path, output_dir, feature_density_source_path),
    ]
    manifest = {path.name: _sha256(path) for path in artifacts}
    (output_dir / "figure_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dark-summary", type=Path, default=DEFAULT_DARK_SUMMARY)
    parser.add_argument(
        "--dense-phase-source",
        type=Path,
        default=DEFAULT_DENSE_PHASE_SOURCE,
    )
    parser.add_argument(
        "--dense-suffix-source",
        type=Path,
        default=DEFAULT_DENSE_SUFFIX_SOURCE,
    )
    parser.add_argument(
        "--dense-n4-phase-source",
        type=Path,
        default=DEFAULT_DENSE_N4_PHASE_SOURCE,
    )
    parser.add_argument(
        "--dense-n4-suffix-source",
        type=Path,
        default=DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    )
    parser.add_argument("--scaling-source", type=Path, default=DEFAULT_SCALING_SOURCE)
    parser.add_argument("--suffix-source", type=Path, default=DEFAULT_SUFFIX_SOURCE)
    parser.add_argument("--shot-source", type=Path, default=DEFAULT_SHOT_SOURCE)
    parser.add_argument("--feature-density-source", type=Path, default=DEFAULT_FEATURE_DENSITY_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = render_all(
        dark_summary_path=args.dark_summary.resolve(),
        dense_phase_source_path=args.dense_phase_source.resolve(),
        dense_suffix_source_path=args.dense_suffix_source.resolve(),
        dense_n4_phase_source_path=args.dense_n4_phase_source.resolve(),
        dense_n4_suffix_source_path=args.dense_n4_suffix_source.resolve(),
        scaling_source_path=args.scaling_source.resolve(),
        suffix_source_path=args.suffix_source.resolve(),
        shot_source_path=args.shot_source.resolve(),
        feature_density_source_path=args.feature_density_source.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
