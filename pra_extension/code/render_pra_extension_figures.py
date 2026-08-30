"""Render publication figures for the post-confirmatory PRA extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


COLORS = {
    "local_loss": "#D97706",
    "collective_loss": "#2F5D8C",
    "dephasing": "#2E7D5B",
    1: "#6A3D9A",
    2: "#2F5D8C",
    4: "#D97706",
}
LABELS = {
    "local_loss": "local loss",
    "collective_loss": "collective loss",
    "dephasing": "dephasing",
    1: r"$B=18$",
    2: r"$B=36$",
    4: r"$B=72$",
}
PANEL_COLORS = {
    "canonical_xyz_w2": "#6A3D9A",
    "coherence_xy_w1": "#E69F00",
    "population_z_w1": "#56B4E9",
}
PANEL_LABELS = {
    "canonical_xyz_w2": r"local $XYZ$, weight $\leq2$",
    "coherence_xy_w1": r"local $XY$",
    "population_z_w1": r"local $Z$",
}
ARCHITECTURE_COLORS = {
    "dense": "#6A3D9A",
    "chain": "#0072B2",
    "ring": "#E69F00",
    "star": "#009E73",
}
ARCHITECTURE_LABELS = {
    "dense": "dense",
    "chain": "chain",
    "ring": "ring",
    "star": "star",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    """Set the compact, unparenthesized panel style used in the main figures."""

    ax.set_title(rf"$\mathbf{{{letter}}}$  {title}", loc="left", pad=4.5, fontsize=9.2)


def _set_print_typography(axes: np.ndarray) -> None:
    """Keep three-panel figures legible at full-width journal print size."""

    for ax in axes:
        ax.xaxis.label.set_size(8.9)
        ax.yaxis.label.set_size(8.9)
        ax.tick_params(axis="both", labelsize=8.0)


def observable_figure(
    results: pd.DataFrame,
    root_effects: pd.DataFrame,
    summary: dict[str, object],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.08))

    # (a) Paired d=1 -> d=8 effects on the same 79 live cells in every family.
    ax = axes[0]
    order = ["local_loss", "collective_loss", "dephasing"]
    rng = np.random.default_rng(28_082_650)
    for xpos, family in enumerate(order):
        values = root_effects.loc[
            root_effects["dissipator_family"].eq(family),
            "common_live_gain_d1_to_d8",
        ].to_numpy(float)
        jitter = rng.uniform(-0.10, 0.10, len(values))
        ax.scatter(
            np.full(len(values), xpos) + jitter,
            100 * values,
            s=17,
            facecolor="white",
            edgecolor=COLORS[family],
            linewidth=0.8,
            alpha=0.9,
            zorder=2,
        )
        effect = summary["observable_memory"]["common_live_support_sensitivity"][
            "effects"
        ][family]["fractional_reduction"]
        median = 100 * effect["median"]
        low = 100 * effect["ci95_low"]
        high = 100 * effect["ci95_high"]
        ax.errorbar(
            xpos,
            median,
            yerr=[[median - low], [high - median]],
            fmt="_",
            markersize=13,
            markeredgewidth=2.2,
            color="black",
            capsize=2.5,
            zorder=3,
        )
        ax.text(
            xpos,
            median + 5.0,
            f"{median:.0f}%",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=COLORS[family],
            fontweight="bold",
        )
    ax.axhline(0, color="0.55", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(3), ["local\nloss", "collective\nloss", "dephasing"])
    ax.set_ylabel(r"History-NRMSE reduction, $d=1\to8$ (\%)")
    ax.set_ylim(-4, 104)
    _panel_title(ax, "a", "Common-support forgetting")

    # (b) Signal liveness exposes collapse rather than useful forgetting.
    ax = axes[1]
    live = results.assign(live=results["target_variance"] >= 1e-8)
    live_fraction = (
        live.groupby(["dissipator_family", "reference_local_loss_gamma"])["live"]
        .mean()
        .rename("fraction")
        .reset_index()
    )
    local_live = live_fraction[
        live_fraction["dissipator_family"].eq("local_loss")
    ].sort_values("reference_local_loss_gamma")
    collective_live = live_fraction[
        live_fraction["dissipator_family"].eq("collective_loss")
    ].sort_values("reference_local_loss_gamma")
    if not np.allclose(local_live["fraction"], collective_live["fraction"]):
        raise ValueError("local and collective liveness no longer coincide")
    ax.plot(
        np.arange(len(local_live)),
        100 * local_live["fraction"],
        marker="o",
        markersize=4,
        color=COLORS["collective_loss"],
        linewidth=1.7,
    )
    dephasing_live = live_fraction[
        live_fraction["dissipator_family"].eq("dephasing")
    ].sort_values("reference_local_loss_gamma")
    ax.plot(
        np.arange(len(dephasing_live)),
        100 * dephasing_live["fraction"],
        marker="o",
        markersize=4,
        color=COLORS["dephasing"],
        linewidth=1.7,
    )
    ax.text(
        1.97,
        97.0,
        "local + collective",
        color=COLORS["collective_loss"],
        ha="right",
        va="top",
        fontsize=6.8,
        fontweight="bold",
    )
    ax.text(
        1.97,
        18.0,
        "dephasing",
        color=COLORS["dephasing"],
        ha="right",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
    )
    ax.set_xticks([0, 1, 2], ["0.25", "1", "4"])
    ax.set_xlabel(r"Matched reference rate $\gamma$")
    ax.set_ylabel(r"Live target panels ($V_y\geq10^{-8}$) (\%)")
    ax.set_ylim(-3, 104)
    _panel_title(ax, "b", "Signal survival")

    # (c) Collective observable history at high matched strength.
    ax = axes[2]
    high = live[
        live["live"]
        & live["reference_local_loss_gamma"].eq(4.0)
        & live["window"].eq(8)
        & live["dissipator_family"].isin(["local_loss", "collective_loss"])
    ]
    root_values = (
        high.groupby(
            ["root_id", "dissipator_family", "teacher_n_qubits", "observable_panel"]
        )["history_nrmse"]
        .median()
        .reset_index()
    )
    family_offsets = {"local_loss": -0.12, "collective_loss": 0.12}
    panels = ["canonical_xyz_w2", "coherence_xy_w1", "population_z_w1"]
    panel_offsets = {
        "canonical_xyz_w2": -0.035,
        "coherence_xy_w1": 0.0,
        "population_z_w1": 0.035,
    }
    panel_markers = {
        "canonical_xyz_w2": "o",
        "coherence_xy_w1": "s",
        "population_z_w1": "^",
    }

    def median_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(0, len(values), size=(20_000, len(values)))]
        boot = np.median(draws, axis=1)
        return (
            float(np.median(values)),
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        )

    family_linestyles = {"local_loss": (0, (3, 2)), "collective_loss": "solid"}
    for family_index, family in enumerate(["local_loss", "collective_loss"]):
        for panel_index, panel in enumerate(panels):
            medians: list[float] = []
            positions: list[float] = []
            for n_qubits in [2, 3, 4]:
                values = root_values.loc[
                    root_values["dissipator_family"].eq(family)
                    & root_values["teacher_n_qubits"].eq(n_qubits)
                    & root_values["observable_panel"].eq(panel),
                    "history_nrmse",
                ].to_numpy(float)
                xpos = n_qubits + family_offsets[family] + panel_offsets[panel]
                median, low, high_ci = median_ci(
                    values, 28_082_660 + 100 * family_index + 10 * panel_index + n_qubits
                )
                ax.scatter(
                    np.full(len(values), xpos),
                    values,
                    s=9,
                    facecolor="white",
                    edgecolor=COLORS[family],
                    marker=panel_markers[panel],
                    alpha=0.22,
                    linewidth=0.5,
                )
                ax.errorbar(
                    xpos,
                    median,
                    yerr=[[median - low], [high_ci - median]],
                    fmt=panel_markers[panel],
                    markersize=4.0,
                    markerfacecolor="white",
                    markeredgecolor=COLORS[family],
                    color=COLORS[family],
                    linewidth=0.85,
                    capsize=1.5,
                    zorder=3,
                )
                positions.append(xpos)
                medians.append(median)
            ax.plot(
                positions,
                medians,
                color=COLORS[family],
                linewidth=0.9,
                linestyle=family_linestyles[family],
                alpha=0.92,
            )
    ax.axhline(0.1, color="0.35", linestyle="--", linewidth=0.8)
    ax.text(4.16, 0.105, r"$\tau=0.10$", fontsize=7.0, ha="right")
    ax.set_yscale("log")
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel(r"Teacher qubits $N_T$")
    ax.set_ylabel(r"History NRMSE ($\gamma=4,d=8$)")
    ax.set_ylim(2e-8, 0.5)
    _panel_title(ax, "c", "Observable-dependent history")
    handles = [
        Line2D([0], [0], color=COLORS["local_loss"], lw=1.5, linestyle=(0, (3, 2)), label="local"),
        Line2D([0], [0], color=COLORS["collective_loss"], lw=1.5, label="collective"),
        Line2D([0], [0], color="0.35", marker="o", markerfacecolor="white", lw=0, label=r"$XYZ$, $w\leq2$"),
        Line2D([0], [0], color="0.35", marker="s", markerfacecolor="white", lw=0, label=r"local $XY$"),
        Line2D([0], [0], color="0.35", marker="^", markerfacecolor="white", lw=0, label=r"local $Z$"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.005),
        fontsize=6.8,
        ncol=5,
        columnspacing=0.9,
        handlelength=1.6,
    )

    _set_print_typography(axes)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.88", linewidth=0.45, alpha=0.65)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.92, bottom=0.27, wspace=0.42)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def shot_figure(
    results: pd.DataFrame,
    summary: dict[str, object],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65))
    finite = results[results["allocation_mode"].eq("optimal")].copy()
    noiseless = results[results["allocation_mode"].eq("noiseless")].copy()
    positions = np.arange(5)
    shot_values = [300, 1200, 4800, 19200]

    # (a) Fixed-B expected error surface and exact-feature endpoint.
    ax = axes[0]
    for budget in [1, 2, 4]:
        medians: list[float] = []
        q25: list[float] = []
        q75: list[float] = []
        for total in shot_values:
            values = finite.loc[
                finite["snapshot_budget"].eq(budget)
                & finite["total_shots"].eq(total),
                "test_expected_total_nrmse",
            ].to_numpy(float)
            medians.append(float(np.median(values)))
            q25.append(float(np.quantile(values, 0.25)))
            q75.append(float(np.quantile(values, 0.75)))
        exact = noiseless.loc[
            noiseless["snapshot_budget"].eq(budget), "test_noiseless_nrmse"
        ].to_numpy(float)
        medians.append(float(np.median(exact)))
        q25.append(float(np.quantile(exact, 0.25)))
        q75.append(float(np.quantile(exact, 0.75)))
        ax.plot(
            positions,
            medians,
            marker="o",
            markersize=4,
            color=COLORS[budget],
            label=LABELS[budget],
        )
        ax.fill_between(positions, q25, q75, color=COLORS[budget], alpha=0.12)
    ax.axhline(0.1, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions, ["300", "1.2k", "4.8k", "19.2k", "exact"])
    ax.set_xlabel(r"Total student shots per input $Q$")
    ax.set_ylabel("Analytic shot-aware test NRMSE")
    ax.set_ylim(0.035, 0.36)
    ax.set_title("(a) Feature--resolution frontier", loc="left")
    ax.legend(frameon=False, ncol=1)

    # (b) Validation-only budget selection.
    ax = axes[1]
    counts = summary["finite_shots"]["validation_selected_budget_counts"]
    count_frame = pd.DataFrame(counts)
    count_frame = count_frame[count_frame["allocation_mode"].eq("optimal")]
    bottoms = np.zeros(len(shot_values), float)
    for budget in [1, 2, 4]:
        values = []
        for total in shot_values:
            match = count_frame[
                count_frame["snapshot_budget"].eq(budget)
                & count_frame["total_shots"].eq(total)
            ]
            values.append(int(match["root_teacher_cells"].iloc[0]) if len(match) else 0)
        fractions = np.asarray(values) / 24.0
        ax.bar(
            np.arange(4),
            fractions,
            bottom=bottoms,
            width=0.72,
            color=COLORS[budget],
            label=LABELS[budget],
        )
        for xpos, (bottom, height, value) in enumerate(zip(bottoms, fractions, values)):
            if height >= 0.10:
                ax.text(
                    xpos,
                    bottom + height / 2,
                    str(value),
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                )
        bottoms += fractions
    ax.set_xticks(np.arange(4), ["300", "1.2k", "4.8k", "19.2k"])
    ax.set_xlabel(r"Total student shots per input $Q$")
    ax.set_ylabel("Validation-selected fraction")
    ax.set_ylim(0, 1)
    ax.set_title("(b) Validation-selected $B$", loc="left")

    # (c) Held-out benefit of target-aware allocation at the root level.
    ax = axes[2]
    wide = results[results["allocation_mode"].ne("noiseless")].pivot(
        index=["root_id", "teacher_n_qubits", "snapshot_budget", "total_shots"],
        columns="allocation_mode",
        values="test_expected_total_nrmse",
    )
    wide["gain"] = 1.0 - wide["optimal"] / wide["equal"]
    root_gain = wide["gain"].groupby(["root_id", "total_shots"]).median()
    rng = np.random.default_rng(28_082_651)
    effects = summary["finite_shots"]["root_block_effects"][
        "optimal_vs_equal_by_Q"
    ]
    for xpos, total in enumerate(shot_values):
        values = root_gain.xs(total, level="total_shots").to_numpy(float)
        ax.scatter(
            np.full(len(values), xpos) + rng.uniform(-0.09, 0.09, len(values)),
            100 * values,
            s=15,
            facecolor="white",
            edgecolor="#333333",
            linewidth=0.7,
        )
        effect = effects[f"Q{total}"]
        median = 100 * effect["median"]
        low = 100 * effect["ci95_low"]
        high = 100 * effect["ci95_high"]
        ax.errorbar(
            xpos,
            median,
            yerr=[[median - low], [high - median]],
            fmt="_",
            markersize=13,
            markeredgewidth=2.2,
            color="#D55E00",
            capsize=2.5,
        )
    ax.axhline(0, color="0.55", linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(4), ["300", "1.2k", "4.8k", "19.2k"])
    ax.set_xlabel(r"Total student shots per input $Q$")
    ax.set_ylabel("Design-optimal vs equal NRMSE gain (%)")
    ax.set_ylim(-1, 23)
    ax.set_title("(c) Design-optimal allocation gain", loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.1)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def n4_shot_robustness_figure(
    results: pd.DataFrame,
    n4_summary: dict[str, object],
    main_summary: dict[str, object],
    output: Path,
) -> None:
    """Show that the finite-feature/finite-shot crossover persists for N_T=4."""

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65))
    finite = results[results["allocation_mode"].eq("optimal")].copy()
    noiseless = results[results["allocation_mode"].eq("noiseless")].copy()
    shot_values = [300, 1200, 4800, 19200]
    positions = np.arange(5)

    # (a) Four-qubit-teacher error surface under target-aware allocation.
    ax = axes[0]
    for budget in [1, 2, 4]:
        medians: list[float] = []
        q25: list[float] = []
        q75: list[float] = []
        for total in shot_values:
            values = finite.loc[
                finite["snapshot_budget"].eq(budget)
                & finite["total_shots"].eq(total),
                "test_expected_total_nrmse",
            ].to_numpy(float)
            medians.append(float(np.median(values)))
            q25.append(float(np.quantile(values, 0.25)))
            q75.append(float(np.quantile(values, 0.75)))
        exact = noiseless.loc[
            noiseless["snapshot_budget"].eq(budget), "test_noiseless_nrmse"
        ].to_numpy(float)
        medians.append(float(np.median(exact)))
        q25.append(float(np.quantile(exact, 0.25)))
        q75.append(float(np.quantile(exact, 0.75)))
        ax.plot(
            positions,
            medians,
            marker="o",
            markersize=4,
            color=COLORS[budget],
            label=LABELS[budget],
        )
        ax.fill_between(positions, q25, q75, color=COLORS[budget], alpha=0.12)
    ax.axhline(0.1, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions, ["300", "1.2k", "4.8k", "19.2k", "exact"])
    ax.set_xlabel(r"Total student shots per input $Q$")
    ax.set_ylabel("Analytic shot-aware test NRMSE")
    ax.set_ylim(0.035, 0.34)
    ax.set_title(r"(a) Four-qubit teacher ($N_T=4$)", loc="left")
    ax.legend(frameon=False)

    # (b) Validation-only budget selection across 12 fresh roots.
    ax = axes[1]
    count_frame = pd.DataFrame(n4_summary["validation_selected_budget_counts"])
    count_frame = count_frame[count_frame["allocation_mode"].eq("optimal")]
    bottoms = np.zeros(len(shot_values), float)
    for budget in [1, 2, 4]:
        counts: list[int] = []
        for total in shot_values:
            match = count_frame[
                count_frame["snapshot_budget"].eq(budget)
                & count_frame["total_shots"].eq(total)
            ]
            counts.append(int(match["roots"].iloc[0]) if len(match) else 0)
        fractions = np.asarray(counts, float) / 12.0
        ax.bar(
            np.arange(4),
            fractions,
            bottom=bottoms,
            width=0.72,
            color=COLORS[budget],
            label=LABELS[budget],
        )
        for xpos, (bottom, height, count) in enumerate(zip(bottoms, fractions, counts)):
            if height >= 0.10:
                ax.text(
                    xpos,
                    bottom + height / 2,
                    str(count),
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                )
        bottoms += fractions
    ax.set_xticks(np.arange(4), ["300", "1.2k", "4.8k", "19.2k"])
    ax.set_xlabel(r"Total student shots per input $Q$")
    ax.set_ylabel("Validation-selected fraction")
    ax.set_ylim(0, 1)
    ax.set_title("(b) Selected feature budget", loc="left")

    # (c) Root-bootstrap comparison with the N_T=2,3 arm.
    ax = axes[2]
    earlier = main_summary["finite_shots"]["root_block_effects"][
        "B72_vs_B18_by_allocation_and_Q"
    ]
    series = [
        (r"$N_T=2,3$", earlier, -0.07, "#333333", "optimal_Q"),
        (r"$N_T=4$", n4_summary["fixed_B72_vs_B18"], 0.07, "#E69F00", "Q"),
    ]
    for label, effects, offset, color, prefix in series:
        medians: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for total in shot_values:
            effect = effects[f"{prefix}{total}"]
            medians.append(100 * float(effect["median"]))
            lows.append(100 * float(effect["ci95_low"]))
            highs.append(100 * float(effect["ci95_high"]))
        x = np.arange(4, dtype=float) + offset
        medians_array = np.asarray(medians)
        ax.errorbar(
            x,
            medians_array,
            yerr=[medians_array - np.asarray(lows), np.asarray(highs) - medians_array],
            marker="o",
            markersize=4,
            color=color,
            capsize=2,
            label=label,
        )
    ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(4), ["300", "1.2k", "4.8k", "19.2k"])
    ax.set_xlabel(r"Total student shots per input $Q$")
    ax.set_ylabel(r"Design-optimal $B=72$ vs $B=18$ gain (\%)")
    ax.set_ylim(-10, 72)
    ax.set_title("(c) Size robustness", loc="left")
    ax.legend(frameon=False, loc="upper left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.1)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def spectral_visibility_figure(
    observable: pd.DataFrame,
    spectrum: pd.DataFrame,
    output: Path,
) -> None:
    """Show why channel spectrum, excitation, and measurement must be separated."""

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65))
    family_order = ["local_loss", "dephasing", "collective_loss"]
    n_values = np.asarray([2, 3, 4])
    offsets = {"local_loss": -0.12, "dephasing": 0.0, "collective_loss": 0.12}

    # (a) Representative-input spectrum.  This is a diagnostic, not a uniform bound.
    ax = axes[0]
    high_spectrum = spectrum[spectrum["reference_local_loss_gamma"].eq(4.0)]
    for family in family_order:
        medians: list[float] = []
        for n_qubits in n_values:
            values = high_spectrum.loc[
                high_spectrum["dissipator_family"].eq(family)
                & high_spectrum["teacher_n_qubits"].eq(n_qubits),
                "spectral_radius_traceless",
            ].to_numpy(float)
            xpos = n_qubits + offsets[family]
            ax.scatter(
                np.full(len(values), xpos),
                values,
                s=9,
                color=COLORS[family],
                alpha=0.25,
                linewidth=0,
            )
            medians.append(float(np.median(values)))
        ax.plot(
            n_values + offsets[family],
            medians,
            marker="o",
            markersize=4,
            color=COLORS[family],
            label=LABELS[family],
        )
    ax.axhline(1.0, color="0.45", linestyle="--", linewidth=0.8)
    ax.set_xticks(n_values)
    ax.set_xlabel(r"Teacher qubits $N_T$")
    ax.set_ylabel(r"Mid-drive traceless spectral radius $\rho(A_{0.5})$")
    ax.set_ylim(0.25, 1.025)
    ax.set_title("(a) Slow channel modes", loc="left")
    ax.legend(frameon=False, loc="lower right")

    # (b) Those modes matter only when the generated histories populate them.
    ax = axes[1]
    high_observable = observable[
        observable["reference_local_loss_gamma"].eq(4.0)
        & observable["window"].eq(8)
    ].drop_duplicates(
        ["root_id", "teacher_n_qubits", "dissipator_family"]
    )
    for family in family_order:
        medians = []
        for n_qubits in n_values:
            values = high_observable.loc[
                high_observable["dissipator_family"].eq(family)
                & high_observable["teacher_n_qubits"].eq(n_qubits),
                "final_state_hs_mse",
            ].to_numpy(float)
            xpos = n_qubits + offsets[family]
            ax.scatter(
                np.full(len(values), xpos),
                values,
                s=9,
                color=COLORS[family],
                alpha=0.25,
                linewidth=0,
            )
            medians.append(float(np.median(values)))
        ax.plot(
            n_values + offsets[family],
            medians,
            marker="o",
            markersize=4,
            color=COLORS[family],
        )
    ax.set_yscale("log")
    ax.set_xticks(n_values)
    ax.set_xlabel(r"Teacher qubits $N_T$")
    ax.set_ylabel(r"Surviving state covariance $V_{\rm HS}(d=8)$")
    ax.set_ylim(7e-17, 4e-4)
    ax.set_title("(b) Excited surviving memory", loc="left")

    # (c) Pauli Parseval gives the exact fraction of surviving state variance
    # exposed by a selected orthogonal Pauli coordinate panel.
    ax = axes[2]
    collective = observable[
        observable["reference_local_loss_gamma"].eq(4.0)
        & observable["window"].eq(8)
        & observable["dissipator_family"].eq("collective_loss")
    ].copy()
    collective["measurement_visibility_fraction"] = (
        collective["n_outputs"]
        * collective["history_mse"]
        / (
            (2.0 ** collective["teacher_n_qubits"])
            * collective["final_state_hs_mse"]
        )
    )
    panel_offsets = {
        "canonical_xyz_w2": -0.10,
        "coherence_xy_w1": 0.0,
        "population_z_w1": 0.10,
    }
    for panel in PANEL_COLORS:
        medians = []
        for n_qubits in n_values:
            values = collective.loc[
                collective["observable_panel"].eq(panel)
                & collective["teacher_n_qubits"].eq(n_qubits),
                "measurement_visibility_fraction",
            ].to_numpy(float)
            xpos = n_qubits + panel_offsets[panel]
            ax.scatter(
                np.full(len(values), xpos),
                values,
                s=9,
                color=PANEL_COLORS[panel],
                alpha=0.25,
                linewidth=0,
            )
            medians.append(float(np.median(values)))
        ax.plot(
            n_values + panel_offsets[panel],
            medians,
            marker="o",
            markersize=4,
            color=PANEL_COLORS[panel],
            label=PANEL_LABELS[panel],
        )
    ax.set_yscale("log")
    ax.set_xticks(n_values)
    ax.set_xlabel(r"Teacher qubits $N_T$")
    ax.set_ylabel("Measured fraction of state covariance")
    ax.set_ylim(0.006, 1.15)
    ax.set_title("(c) Measurement visibility", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=6.4)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.15)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def continuous_robustness_figure(
    quantized: pd.DataFrame,
    continuous: pd.DataFrame,
    continuous_roots: pd.DataFrame,
    output: Path,
) -> None:
    """Compare the matched strong-loss mechanism on gridded and continuous inputs."""

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65))
    families = ["local_loss", "collective_loss", "dephasing"]

    # (a) Continuous-input paired endpoints, balanced inside roots.
    ax = axes[0]
    for family in families:
        group = continuous_roots[
            continuous_roots["dissipator_family"].eq(family)
        ]
        for row in group.itertuples(index=False):
            ax.plot(
                [1, 8],
                [row.median_history_nrmse_d1, row.median_history_nrmse_d8],
                color=COLORS[family],
                alpha=0.18,
                linewidth=0.7,
            )
        ax.plot(
            [1, 8],
            [
                float(np.median(group["median_history_nrmse_d1"])),
                float(np.median(group["median_history_nrmse_d8"])),
            ],
            color=COLORS[family],
            marker="o",
            markersize=4,
            label=LABELS[family],
        )
    ax.set_yscale("log")
    ax.set_xticks([1, 8])
    ax.set_xlabel("Continuous-input window $d$")
    ax.set_ylabel("Root-median history NRMSE")
    ax.set_ylim(8e-5, 1.2)
    ax.set_title("(a) Continuous endpoints", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=6.7)

    # (b) Same matched gamma and teacher sizes across the two input ensembles.
    ax = axes[1]
    quantized_high = quantized[
        quantized["reference_local_loss_gamma"].eq(4.0)
        & quantized["teacher_n_qubits"].isin([2, 3])
    ]
    wide = quantized_high.pivot(
        index=["root_id", "teacher_n_qubits", "dissipator_family", "observable_panel"],
        columns="window",
        values=["history_nrmse", "target_variance"],
    )
    wide = wide[
        (wide[("target_variance", 1)] >= 1e-8)
        & (wide[("target_variance", 8)] >= 1e-8)
    ]
    quantized_gain = pd.DataFrame(
        {"gain": 1.0 - wide[("history_nrmse", 8)] / wide[("history_nrmse", 1)]}
    ).reset_index()
    quantized_root = (
        quantized_gain.groupby(["root_id", "dissipator_family"])["gain"]
        .median()
        .reset_index()
    )
    rng = np.random.default_rng(28_082_652)
    sources = [
        ("13-symbol grid", quantized_root, "gain", -0.13, "white", "#333333"),
        (
            "continuous",
            continuous_roots,
            "gain_d1_to_d8",
            0.13,
            "#E69F00",
            "#E69F00",
        ),
    ]
    for label, frame, column, offset, face, edge in sources:
        for xpos, family in enumerate(families):
            values = frame.loc[frame["dissipator_family"].eq(family), column].to_numpy(
                float
            )
            jitter = rng.uniform(-0.045, 0.045, len(values))
            ax.scatter(
                np.full(len(values), xpos + offset) + jitter,
                100 * values,
                s=13,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.65,
                alpha=0.85,
                label=label if xpos == 0 else None,
            )
            ax.plot(
                [xpos + offset - 0.065, xpos + offset + 0.065],
                [100 * np.median(values)] * 2,
                color="black",
                linewidth=1.5,
            )
    ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(3), ["local\nloss", "collective\nloss", "dephasing"])
    ax.set_ylabel(r"History-NRMSE reduction, $d=1\to8$ (\%)")
    ax.set_ylim(-8, 106)
    ax.set_title("(b) Alphabet robustness", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=6.7)

    # (c) Liveness changes quantitatively even when the ordering is stable.
    ax = axes[2]
    quantized_live = quantized_high.assign(
        live=quantized_high["target_variance"] >= 1e-8
    ).groupby("dissipator_family")["live"].mean()
    continuous_live = continuous.assign(
        live=continuous["target_variance"] >= 1e-8
    ).groupby("dissipator_family")["live"].mean()
    x = np.arange(3)
    width = 0.34
    first = 100 * np.asarray([quantized_live[family] for family in families])
    second = 100 * np.asarray([continuous_live[family] for family in families])
    ax.bar(x - width / 2, first, width, color="white", edgecolor="#333333")
    ax.bar(x + width / 2, second, width, color="#E69F00", edgecolor="#E69F00")
    for xpos, values in zip(x, zip(first, second)):
        for shift, value in zip((-width / 2, width / 2), values):
            ax.text(xpos + shift, value + 2.0, f"{value:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, ["local\nloss", "collective\nloss", "dephasing"])
    ax.set_ylabel(r"Rows above $V_y=10^{-8}$ (\%)")
    ax.set_ylim(0, 112)
    ax.set_title("(c) Ensemble-dependent liveness", loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.15)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def validate_architecture_figure_inputs(
    results: pd.DataFrame,
    selected_effects: pd.DataFrame,
    analysis: dict[str, object],
) -> None:
    """Fail closed unless raw rows and strict-analysis inputs agree exactly."""

    def root_bootstrap_ci(values: np.ndarray, seed: int) -> np.ndarray:
        array = np.asarray(values, float)
        rng = np.random.default_rng(int(seed))
        draws = rng.choice(array, size=(50_000, len(array)), replace=True)
        return np.quantile(np.median(draws, axis=1), [0.025, 0.975])

    required_raw = {
        "root_id",
        "teacher_n_qubits",
        "student_model",
        "student_architecture",
        "snapshot_budget",
        "validation_nrmse",
        "test_nrmse",
    }
    required_effects = {
        "root_id",
        "teacher_n_qubits",
        "selected_architecture",
        "selected_validation_nrmse",
        "selected_qelm_test_nrmse",
        "quadratic_validation_nrmse",
        "quadratic_test_nrmse",
        "selected_qelm_gain_over_quadratic_percent",
    }
    if required_raw - set(results.columns):
        raise ValueError("architecture raw results are missing required columns")
    if required_effects - set(selected_effects.columns):
        raise ValueError("architecture selected-effects table is incomplete")
    if analysis.get("status") != "EXPLORATORY_STRESS_TEST":
        raise ValueError("architecture analysis status is not exploratory")
    counts = analysis.get("counts", {})
    if counts != {"root_teacher_cells": 36, "roots": 12, "rows": 540}:
        raise ValueError("architecture analysis does not have the full declared profile")
    if len(results) != 540 or results["root_id"].nunique() != 12:
        raise ValueError("architecture raw results do not have the full declared profile")

    qelm_rows = results[results["student_model"].eq("qelm")]
    if set(qelm_rows["student_architecture"]) != set(ARCHITECTURE_COLORS):
        raise ValueError("architecture raw results have the wrong QELM topology family")
    budgets = sorted(int(value) for value in qelm_rows["snapshot_budget"].unique())
    if budgets != [1, 2, 4]:
        raise ValueError("architecture raw results have the wrong feature budgets")
    raw_keys = [
        "root_id",
        "teacher_n_qubits",
        "snapshot_budget",
        "student_architecture",
    ]
    if results.duplicated(raw_keys).any():
        raise ValueError("architecture raw results contain duplicate cells")

    # Reconstruct the validation-only topology choice from raw rows.  No test
    # value participates in sorting or selection.
    high = qelm_rows[qelm_rows["snapshot_budget"].eq(4)].sort_values(
        ["root_id", "teacher_n_qubits", "validation_nrmse", "student_architecture"],
        kind="mergesort",
    )
    selected = high.groupby(["root_id", "teacher_n_qubits"], sort=True).head(1)
    control = results[
        results["student_model"].eq("quadratic_delay")
        & results["snapshot_budget"].eq(4)
    ]
    reconstructed = selected[
        [
            "root_id",
            "teacher_n_qubits",
            "student_architecture",
            "validation_nrmse",
            "test_nrmse",
        ]
    ].rename(
        columns={
            "student_architecture": "selected_architecture",
            "validation_nrmse": "selected_validation_nrmse",
            "test_nrmse": "selected_qelm_test_nrmse",
        }
    ).merge(
        control[
            ["root_id", "teacher_n_qubits", "validation_nrmse", "test_nrmse"]
        ].rename(
            columns={
                "validation_nrmse": "quadratic_validation_nrmse",
                "test_nrmse": "quadratic_test_nrmse",
            }
        ),
        on=["root_id", "teacher_n_qubits"],
        validate="one_to_one",
    )
    reconstructed["selected_qelm_gain_over_quadratic_percent"] = 100.0 * (
        reconstructed["quadratic_test_nrmse"]
        - reconstructed["selected_qelm_test_nrmse"]
    ) / reconstructed["quadratic_test_nrmse"]
    ordered = ["root_id", "teacher_n_qubits"]
    expected = reconstructed.sort_values(ordered, kind="mergesort").reset_index(drop=True)
    observed = selected_effects.sort_values(ordered, kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        observed[expected.columns], expected, check_exact=False, atol=2e-14, rtol=2e-14
    )

    # The root-block feature effects printed in panel (a) must be those in the
    # strict JSON, not a fresh unregistered summary invented by the renderer.
    pivot = qelm_rows.pivot(
        index=["root_id", "teacher_n_qubits", "student_architecture"],
        columns="snapshot_budget",
        values="test_nrmse",
    )
    pivot["gain_percent"] = 100.0 * (pivot[1] - pivot[4]) / pivot[1]
    root_gain = pivot["gain_percent"].groupby(
        ["root_id", "student_architecture"]
    ).median()
    reported = analysis.get("architecture_feature_effects", {})
    for index, architecture in enumerate(ARCHITECTURE_COLORS):
        values = root_gain.xs(architecture, level="student_architecture").to_numpy(float)
        effect = reported.get(architecture, {})
        if not np.isclose(
            np.median(values),
            float(effect.get("median_root_feature_gain_percent", np.nan)),
            atol=2e-13,
            rtol=2e-13,
        ):
            raise ValueError("architecture root effect disagrees with strict analysis")
        if int(effect.get("positive_roots", -1)) != int(np.sum(values > 0.0)):
            raise ValueError("architecture positive-root count disagrees with raw rows")
        cells = pivot.xs(architecture, level="student_architecture")["gain_percent"]
        if int(effect.get("positive_root_teacher_cells", -1)) != int(
            np.sum(cells.to_numpy(float) > 0.0)
        ):
            raise ValueError("architecture positive-cell count disagrees with raw rows")
        if not np.allclose(
            root_bootstrap_ci(values, 86_100 + index),
            np.asarray(effect.get("bootstrap_95_ci_percent", []), float),
            atol=2e-13,
            rtol=2e-13,
        ):
            raise ValueError("architecture bootstrap interval disagrees with raw rows")

    comparison = analysis.get("validation_selected_qelm_vs_quadratic", {})
    root_comparison = (
        observed.groupby("root_id", sort=True)[
            "selected_qelm_gain_over_quadratic_percent"
        ]
        .median()
        .to_numpy(float)
    )
    if not np.isclose(
        np.median(root_comparison),
        float(comparison.get("median_root_gain_percent", np.nan)),
        atol=2e-13,
        rtol=2e-13,
    ):
        raise ValueError("selected-QELM root effect disagrees with strict analysis")
    if not np.allclose(
        root_bootstrap_ci(root_comparison, 86_200),
        np.asarray(comparison.get("bootstrap_95_ci_percent", []), float),
        atol=2e-13,
        rtol=2e-13,
    ):
        raise ValueError("selected-QELM bootstrap interval disagrees with raw rows")
    positive_cells = int(
        np.sum(observed["selected_qelm_gain_over_quadratic_percent"].to_numpy(float) > 0)
    )
    if positive_cells != int(comparison.get("positive_root_teacher_cells", -1)):
        raise ValueError("selected-QELM positive-cell count disagrees with raw rows")
    selected_counts = {
        str(name): int(count)
        for name, count in observed["selected_architecture"].value_counts().sort_index().items()
    }
    if selected_counts != analysis.get("validation_selected_architecture_counts"):
        raise ValueError("validation-selected topology counts disagree with raw rows")
    high_wide = high.pivot(
        index=["root_id", "teacher_n_qubits"],
        columns="student_architecture",
        values="test_nrmse",
    )
    control_indexed = control.set_index(["root_id", "teacher_n_qubits"])
    control_beats_all = int(
        np.sum(control_indexed["test_nrmse"] < high_wide.min(axis=1))
    )
    if control_beats_all != int(
        comparison.get("quadratic_beats_every_fixed_qelm_topology_cells", -1)
    ):
        raise ValueError("fixed-topology classical comparison disagrees with raw rows")


def student_architecture_figure(
    results: pd.DataFrame,
    selected_effects: pd.DataFrame,
    analysis: dict[str, object],
    output: Path,
) -> None:
    """Show topology-robust feature gain and the retained classical control."""

    validate_architecture_figure_inputs(results, selected_effects, analysis)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.0))
    architecture_order = ["dense", "chain", "ring", "star"]

    # (a) Every point is one fresh root, balanced by taking the median over the
    # three teacher sizes before plotting.  The intervals are read from the
    # strict root-bootstrap analysis.
    ax = axes[0]
    qelm_rows = results[results["student_model"].eq("qelm")]
    pivot = qelm_rows.pivot(
        index=["root_id", "teacher_n_qubits", "student_architecture"],
        columns="snapshot_budget",
        values="test_nrmse",
    )
    pivot["gain_percent"] = 100.0 * (pivot[1] - pivot[4]) / pivot[1]
    root_gain = pivot["gain_percent"].groupby(
        ["root_id", "student_architecture"]
    ).median()
    rng = np.random.default_rng(28_082_653)
    effects = analysis["architecture_feature_effects"]
    for xpos, architecture in enumerate(architecture_order):
        values = root_gain.xs(architecture, level="student_architecture").to_numpy(float)
        ax.scatter(
            np.full(len(values), xpos) + rng.uniform(-0.09, 0.09, len(values)),
            values,
            s=18,
            facecolor="white",
            edgecolor=ARCHITECTURE_COLORS[architecture],
            linewidth=0.9,
            zorder=2,
        )
        effect = effects[architecture]
        median = float(effect["median_root_feature_gain_percent"])
        low, high = (float(value) for value in effect["bootstrap_95_ci_percent"])
        ax.errorbar(
            xpos,
            median,
            yerr=[[median - low], [high - median]],
            fmt="_",
            markersize=15,
            markeredgewidth=2.4,
            color="black",
            capsize=2.5,
            zorder=3,
        )
        ax.text(
            xpos,
            5.0,
            f"{int(effect['positive_root_teacher_cells'])}/36",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=ARCHITECTURE_COLORS[architecture],
        )
    ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(4), [ARCHITECTURE_LABELS[name] for name in architecture_order])
    ax.set_ylabel(r"Held-out NRMSE reduction, $B=18\to72$ (\%)")
    ax.set_ylim(-4, 101)
    ax.set_title("(a) Feature gain survives topology changes", loc="left")
    ax.text(
        0.02,
        0.97,
        "dots: 12 roots; lines: median + 95% CI\nlabels: positive root×teacher cells",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
    )

    # (b) Architecture is selected by validation NRMSE.  Test NRMSE is only
    # plotted after selection; every point above the diagonal favors quadratic
    # delay.  Color records which topology validation selected.
    ax = axes[1]
    for architecture in architecture_order:
        group = selected_effects[
            selected_effects["selected_architecture"].eq(architecture)
        ]
        if group.empty:
            continue
        ax.scatter(
            group["quadratic_test_nrmse"],
            group["selected_qelm_test_nrmse"],
            s=25,
            facecolor=ARCHITECTURE_COLORS[architecture],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            label=ARCHITECTURE_LABELS[architecture],
            zorder=2,
        )
    all_values = np.concatenate(
        [
            selected_effects["quadratic_test_nrmse"].to_numpy(float),
            selected_effects["selected_qelm_test_nrmse"].to_numpy(float),
        ]
    )
    lower = max(0.0, float(np.min(all_values)) - 0.004)
    upper = float(np.max(all_values)) + 0.004
    ax.plot([lower, upper], [lower, upper], color="0.35", linestyle="--", linewidth=0.9)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Quadratic-delay test NRMSE")
    ax.set_ylabel("Validation-selected QELM test NRMSE")
    ax.set_title("(b) Classical control remains stronger", loc="left")
    comparison = analysis["validation_selected_qelm_vs_quadratic"]
    ax.text(
        0.04,
        0.96,
        "QELM wins: "
        f"{int(comparison['positive_root_teacher_cells'])}/"
        f"{int(comparison['n_root_teacher_cells'])}\n"
        "topology chosen on validation only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
    )
    ax.legend(frameon=False, loc="lower right", title="selected topology", fontsize=6.8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def validate_eiv_figure_inputs(
    results: pd.DataFrame,
    root_effects: pd.DataFrame,
    selected_rows: pd.DataFrame,
    analysis: dict[str, object],
) -> None:
    """Re-run the strict EIV reduction and compare every plotted artifact."""

    # Import locally so the established Figures 6--11 remain usable as module
    # functions without eagerly loading the EIV experiment implementation.
    from analyze_eiv_shots import analyze_eiv

    recomputed_roots, recomputed_selected, recomputed_summary = analyze_eiv(results)
    root_order = ["root_id", "total_shots"]
    observed_roots = root_effects.sort_values(root_order, kind="mergesort").reset_index(
        drop=True
    )
    expected_roots = recomputed_roots.sort_values(
        root_order, kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        observed_roots[expected_roots.columns],
        expected_roots,
        check_dtype=False,
        check_exact=False,
        atol=3e-14,
        rtol=3e-14,
    )

    selected_order = ["root_id", "total_shots", "replicate"]
    observed_selected = selected_rows.sort_values(
        selected_order, kind="mergesort"
    ).reset_index(drop=True)
    expected_selected = recomputed_selected.sort_values(
        selected_order, kind="mergesort"
    ).reset_index(drop=True)
    if list(observed_selected.columns) != list(expected_selected.columns):
        raise ValueError("EIV selected-row columns differ from strict reduction")
    pd.testing.assert_frame_equal(
        observed_selected,
        expected_selected,
        check_dtype=False,
        check_exact=False,
        atol=3e-14,
        rtol=3e-14,
    )

    # JSON-normalized equality catches any changed count, root interval,
    # selection frequency, or policy score while tolerating no substitutions.
    for key, expected in recomputed_summary.items():
        observed = analysis.get(key)
        if json.dumps(observed, sort_keys=True, allow_nan=False) != json.dumps(
            expected, sort_keys=True, allow_nan=False
        ):
            raise ValueError(f"EIV strict-analysis field {key!r} disagrees with raw rows")


def validate_eiv_artifact_hashes(
    *,
    results_path: Path,
    root_effects_path: Path,
    selected_rows_path: Path,
    analysis: dict[str, object],
) -> None:
    """Check that supplied EIV files are the content-addressed strict bundle."""

    source_hashes = analysis.get("source_artifacts", {})
    source_matches = [
        digest
        for name, digest in source_hashes.items()
        if Path(name).name == results_path.name
    ]
    if source_matches != [_sha256(results_path)]:
        raise ValueError("EIV raw CSV does not match the strict analysis source hash")
    artifact_hashes = analysis.get("artifacts", {})
    for path in (root_effects_path, selected_rows_path):
        if artifact_hashes.get(path.name) != _sha256(path):
            raise ValueError(f"EIV strict artifact hash mismatch for {path.name}")


def eiv_all_splits_figure(
    results: pd.DataFrame,
    root_effects: pd.DataFrame,
    selected_rows: pd.DataFrame,
    analysis: dict[str, object],
    output: Path,
) -> None:
    """Render the all-splits-noisy N=4 operational shot frontier."""

    validate_eiv_figure_inputs(results, root_effects, selected_rows, analysis)
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.98))
    shot_values = [300, 1_200, 4_800, 19_200]
    x = np.arange(len(shot_values))

    # (a) Descriptive validation-selection counts over 12 roots x 16 fresh
    # shot-noise replicates.  No test quantity enters the choice of B.
    ax = axes[0]
    count_frame = pd.DataFrame(analysis["validation_selected_budget_counts"])
    bottoms = np.zeros(len(shot_values), float)
    budget_hatches = {1: "....", 2: "////", 4: None}
    for budget in [1, 2, 4]:
        counts: list[int] = []
        for total in shot_values:
            match = count_frame[
                count_frame["snapshot_budget"].eq(budget)
                & count_frame["total_shots"].eq(total)
            ]
            counts.append(int(match["root_replicates"].iloc[0]) if len(match) else 0)
        fractions = np.asarray(counts, float) / 192.0
        ax.bar(
            x,
            fractions,
            bottom=bottoms,
            width=0.72,
            color=COLORS[budget],
            edgecolor="white",
            linewidth=0.35,
            hatch=budget_hatches[budget],
            label=LABELS[budget],
        )
        feature_budget = {1: 18, 2: 36, 4: 72}[budget]
        for xpos, (bottom, height, count) in enumerate(zip(bottoms, fractions, counts)):
            if height >= 0.10 or xpos == 0:
                label = (
                    f"$B={feature_budget}$\n{count}"
                    if xpos in (0, len(x) - 1)
                    else str(count)
                )
                ax.text(
                    xpos,
                    bottom + height / 2,
                    label,
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    fontweight="bold",
                )
        bottoms += fractions
    ax.set_xticks(x, ["300", "1.2k", "4.8k", "19.2k"])
    ax.set_xlabel(r"Shots per input $Q$")
    ax.set_ylabel("Validation selection share")
    ax.set_ylim(0, 1)
    _panel_title(ax, "a", "Validation selects richer features")

    # (b) Each gray line is one root after the 16 replicate scores are reduced
    # within root.  Black intervals are the strict 12-root bootstrap summaries.
    ax = axes[1]
    wide = root_effects.pivot(
        index="root_id", columns="total_shots", values="selected_policy_test_nrmse"
    )
    for _, row in wide.iterrows():
        ax.plot(
            x,
            [row[total] for total in shot_values],
            color="0.68",
            linewidth=0.65,
            alpha=0.55,
            zorder=1,
        )
    fixed_surface = pd.DataFrame(analysis["fixed_test_surface"])
    for budget in [1, 2, 4]:
        curve = fixed_surface[fixed_surface["snapshot_budget"].eq(budget)].sort_values(
            "total_shots"
        )
        ax.plot(
            x,
            curve["median_test_nrmse"],
            color=COLORS[budget],
            linestyle=(0, (3, 2)),
            marker="o",
            markersize=3.2,
            linewidth=1.0,
            alpha=0.88,
            label=LABELS[budget],
            zorder=2,
        )
    effects = analysis["validation_selected_policy_test_nrmse"]
    medians = np.asarray([float(effects[f"Q{total}"]["median"]) for total in shot_values])
    lows = np.asarray([float(effects[f"Q{total}"]["ci95_low"]) for total in shot_values])
    highs = np.asarray([float(effects[f"Q{total}"]["ci95_high"]) for total in shot_values])
    ax.errorbar(
        x,
        medians,
        yerr=[medians - lows, highs - medians],
        color="#20262E",
        marker="o",
        markersize=4.2,
        capsize=2.5,
        linewidth=1.7,
        zorder=3,
        label="validation-selected",
    )
    ax.set_xticks(x, ["300", "1.2k", "4.8k", "19.2k"])
    ax.set_xlabel(r"Shots per input $Q$")
    ax.set_ylabel("Test NRMSE")
    ax.set_ylim(0.055, 0.33)
    _panel_title(ax, "b", "Held-out test error")
    ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=6.6,
        ncol=2,
        columnspacing=0.8,
        handlelength=1.8,
    )
    shot_gain = analysis["validation_selected_policy_Q19200_vs_Q300"]
    ax.text(
        0.97,
        0.96,
        r"$Q=300\to19.2$k: "
        f"{100 * float(shot_gain['median']):.1f}% lower\n"
        f"95% CI [{100 * float(shot_gain['ci95_low']):.1f}, "
        f"{100 * float(shot_gain['ci95_high']):.1f}]%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )

    # (c) The fixed-B=72 all-splits noisy score is divided by the corresponding
    # exact-feature reference for each root.  A value above one is an operational
    # penalty from noisy fitting/selection plus noisy deployment.
    ax = axes[2]
    b72 = results[results["snapshot_budget"].eq(4)]
    noisy = b72.groupby(["root_id", "total_shots"], sort=True)[
        "test_noisy_feature_nrmse"
    ].median()
    reference = b72.groupby("root_id", sort=True)[
        "noiseless_reference_test_nrmse"
    ].first()
    ratios = noisy / reference
    rng = np.random.default_rng(28_082_654)
    degradation = analysis[
        "B72_all_split_noise_fractional_excess_over_noiseless_reference"
    ]
    for xpos, total in enumerate(shot_values):
        values = ratios.xs(total, level="total_shots").to_numpy(float)
        ax.scatter(
            np.full(len(values), xpos) + rng.uniform(-0.08, 0.08, len(values)),
            values,
            s=15,
            facecolor="white",
            edgecolor="#2F5D8C",
            linewidth=0.75,
            zorder=2,
        )
        effect = degradation[f"Q{total}"]
        median = 1.0 + float(effect["median"])
        low = 1.0 + float(effect["ci95_low"])
        high = 1.0 + float(effect["ci95_high"])
        ax.errorbar(
            xpos,
            median,
            yerr=[[median - low], [high - median]],
            fmt="_",
            markersize=13,
            markeredgewidth=2.2,
            color="black",
            capsize=2.5,
            zorder=3,
        )
    ax.axhspan(1.0, 5.4, color="#2F5D8C", alpha=0.025, zorder=0)
    ax.axhline(1.0, color="0.45", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 5], ["1", "1.5", "2", "3", "5"])
    ax.minorticks_off()
    ax.set_ylim(0.96, 5.4)
    ax.set_xticks(x, ["300", "1.2k", "4.8k", "19.2k"])
    ax.set_xlabel(r"Shots per input $Q$")
    ax.set_ylabel(r"$B=72$ noisy/noiseless NRMSE")
    _panel_title(ax, "c", "Distance from noiseless")
    ax.text(
        0.97,
        0.96,
        "dots: 12 roots\nlines: median + 95% CI",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
    )
    ax.text(
        0.03,
        0.04,
        "noiseless limit",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="0.40",
    )

    _set_print_typography(axes)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.88", linewidth=0.45, alpha=0.65)
    fig.tight_layout(w_pad=1.15)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observable", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--observable-root-effects", type=Path, required=True)
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--continuous-root-effects", type=Path, required=True)
    parser.add_argument("--n4-shots", type=Path, required=True)
    parser.add_argument("--n4-summary", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--architecture-analysis", type=Path, required=True)
    parser.add_argument("--architecture-selected-effects", type=Path, required=True)
    parser.add_argument("--eiv-shots", type=Path, required=True)
    parser.add_argument("--eiv-analysis", type=Path, required=True)
    parser.add_argument("--eiv-root-effects", type=Path, required=True)
    parser.add_argument("--eiv-selected-rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    _style()
    observable = pd.read_csv(args.observable)
    shots = pd.read_csv(args.shots)
    root_effects = pd.read_csv(args.observable_root_effects)
    spectrum = pd.read_csv(args.spectrum)
    continuous = pd.read_csv(args.continuous)
    continuous_roots = pd.read_csv(args.continuous_root_effects)
    n4_shots = pd.read_csv(args.n4_shots)
    n4_summary = json.loads(args.n4_summary.read_text())
    architecture = pd.read_csv(args.architecture)
    architecture_analysis = json.loads(args.architecture_analysis.read_text())
    if architecture_analysis.get("source_results_sha256") != _sha256(args.architecture):
        raise ValueError("architecture analysis source hash does not match raw results")
    architecture_selected = pd.read_csv(args.architecture_selected_effects)
    eiv_shots = pd.read_csv(args.eiv_shots)
    eiv_analysis = json.loads(args.eiv_analysis.read_text())
    eiv_root_effects = pd.read_csv(args.eiv_root_effects)
    eiv_selected_rows = pd.read_csv(args.eiv_selected_rows)
    validate_eiv_artifact_hashes(
        results_path=args.eiv_shots,
        root_effects_path=args.eiv_root_effects,
        selected_rows_path=args.eiv_selected_rows,
        analysis=eiv_analysis,
    )
    summary = json.loads(args.summary.read_text())
    observable_figure(
        observable, root_effects, summary, output / "fig06_observable_memory"
    )
    shot_figure(shots, summary, output / "fig07_finite_shot_frontier")
    spectral_visibility_figure(
        observable, spectrum, output / "fig08_spectral_visibility"
    )
    continuous_robustness_figure(
        observable,
        continuous,
        continuous_roots,
        output / "fig09_continuous_robustness",
    )
    n4_shot_robustness_figure(
        n4_shots,
        n4_summary,
        summary,
        output / "fig10_n4_shot_robustness",
    )
    student_architecture_figure(
        architecture,
        architecture_selected,
        architecture_analysis,
        output / "fig11_student_architecture_robustness",
    )
    eiv_all_splits_figure(
        eiv_shots,
        eiv_root_effects,
        eiv_selected_rows,
        eiv_analysis,
        output / "fig12_noisy_training_frontier",
    )
    artifacts = sorted(
        path
        for path in output.glob("fig*.*")
        if path.name != "figure_sha256.json"
    )
    manifest = {path.name: _sha256(path) for path in artifacts}
    (output / "figure_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
