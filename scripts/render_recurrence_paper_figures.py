"""Render paper figures from the sealed recurrence-fingerprint artifacts.

This is a post-held-out presentation script.  It reads the immutable study
outputs and frozen block metadata, writes auditable source tables, and never
re-runs or modifies the registered experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results_recurrence_fingerprint_heldout"
DEFAULT_CONFIG = ROOT / "data" / "frozen" / "recurrence_fingerprint_v1.json"
DEFAULT_OUTPUT = ROOT / "paper" / "figures"

DISSIPATOR_ORDER = ("local_loss", "collective_loss", "dephasing")
DISSIPATOR_LABEL = {
    "local_loss": "Local loss",
    "collective_loss": "Collective loss",
    "dephasing": "Dephasing",
}
DISSIPATOR_COLOR = {
    "local_loss": "#0f766e",
    "collective_loss": "#2563eb",
    "dephasing": "#d97706",
}
SOURCE_ORDER = (
    "qrc",
    "replay_qelm_d1",
    "replay_qelm_d3",
    "replay_qelm_d5",
    "spatial_qelm_dmin",
)
SOURCE_LABEL = {
    "qrc": "QRC",
    "replay_qelm_d1": "Replay\nQELM d=1",
    "replay_qelm_d3": "Replay\nQELM d=3",
    "replay_qelm_d5": "Replay\nQELM d=5",
    "spatial_qelm_dmin": "Spatial\nQELM",
}


@dataclass(frozen=True)
class Block:
    physics_id: str
    n_qubits: int
    dissipator: str
    gamma: float
    dt: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_blocks(path: Path) -> dict[str, Block]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    blocks = {
        row["physics_id"]: Block(
            physics_id=row["physics_id"],
            n_qubits=int(row["n_qubits"]),
            dissipator=row["dissipator"],
            gamma=float(row["gamma"]),
            dt=float(row["dt"]),
        )
        for row in payload["heldout_blocks"]
    }
    if len(blocks) != 24:
        raise ValueError(f"expected 24 held-out blocks, found {len(blocks)}")
    return blocks


def fingerprint_source_tables(
    results: Path,
    blocks: dict[str, Block],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    predictions = _read_csv(results / "fingerprint_predictions_unblinded.csv")
    baselines: list[dict[str, object]] = []
    for row in predictions:
        if not (
            row["source_model"] == "qrc"
            and row["shots_per_batch"] == "infinity"
            and row["measurement_seed"] == "0"
        ):
            continue
        block = blocks[row["physics_id"]]
        baselines.append(
            {
                "physics_id": block.physics_id,
                "n_qubits": block.n_qubits,
                "dissipator": block.dissipator,
                "gamma": block.gamma,
                "dt": block.dt,
                "exact_baseline_signal_mse": float(row["baseline_signal_mse"]),
                "classification": row["classification"],
                "below_registered_floor": float(row["baseline_signal_mse"]) < 1e-4,
            }
        )
    baselines.sort(key=lambda row: str(row["physics_id"]))
    if len(baselines) != 24:
        raise ValueError(f"expected 24 exact QRC baselines, found {len(baselines)}")

    counts: dict[str, Counter[str]] = {source: Counter() for source in SOURCE_ORDER}
    for row in predictions:
        if not (
            row["shots_per_batch"] == "10000"
            and row["primary_case"] == "True"
            and row["source_model"] in counts
        ):
            continue
        label = row["classification"]
        if label.startswith("INDETERMINATE"):
            outcome = "abstain"
        elif row["source_model"] == "qrc" and label == "RECURRENT_RESET_SENSITIVE":
            outcome = "correct"
        elif row["source_model"] != "qrc" and label.startswith("FINITE_WINDOW_COMPATIBLE"):
            inferred = int(row["inferred_cutoff"])
            actual = int(row["actual_window"])
            outcome = "correct" if abs(inferred - actual) <= 1 else "wrong"
        else:
            outcome = "wrong"
        counts[row["source_model"]][outcome] += 1

    summary: list[dict[str, object]] = []
    for source in SOURCE_ORDER:
        n = sum(counts[source].values())
        if n != 192:
            raise ValueError(f"expected 192 primary 10k predictions for {source}, found {n}")
        summary.append(
            {
                "source_model": source,
                "n": n,
                "correct": counts[source]["correct"],
                "abstain": counts[source]["abstain"],
                "wrong": counts[source]["wrong"],
                "correct_rate": counts[source]["correct"] / n,
                "abstention_rate": counts[source]["abstain"] / n,
                "wrong_rate": counts[source]["wrong"] / n,
            }
        )
    return baselines, summary


def render_fingerprint_visibility(
    baselines: list[dict[str, object]],
    summary: list[dict[str, object]],
    output: Path,
) -> None:
    fig, (ax_signal, ax_class) = plt.subplots(
        1, 2, figsize=(10.2, 4.15), gridspec_kw={"width_ratios": [0.92, 1.28]},
        constrained_layout=True,
    )

    for x, dissipator in enumerate(DISSIPATOR_ORDER):
        rows = [row for row in baselines if row["dissipator"] == dissipator]
        rows.sort(key=lambda row: str(row["physics_id"]))
        offsets = np.linspace(-0.16, 0.16, len(rows))
        y = np.asarray([float(row["exact_baseline_signal_mse"]) for row in rows])
        ax_signal.scatter(
            x + offsets,
            y,
            s=43,
            color=DISSIPATOR_COLOR[dissipator],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label=DISSIPATOR_LABEL[dissipator],
        )
        below = sum(bool(row["below_registered_floor"]) for row in rows)
        ax_signal.text(x, 2.0e-8, f"{below}/{len(rows)} below floor", ha="center", va="bottom", fontsize=8)
    ax_signal.axhline(1e-4, color="#991b1b", linestyle="--", linewidth=1.2)
    ax_signal.text(2.42, 1.18e-4, r"registered liveness floor $10^{-4}$", ha="right", va="bottom", fontsize=8, color="#7f1d1d")
    ax_signal.set_yscale("log")
    ax_signal.set_ylim(1e-8, 0.35)
    ax_signal.set_xticks(range(3), [DISSIPATOR_LABEL[d] for d in DISSIPATOR_ORDER], rotation=12)
    ax_signal.set_ylabel(r"Exact paired-history signal $D_{\rm carry}(0)$")
    ax_signal.set_title("a  Memory visibility is observable-dependent", loc="left", fontweight="bold")
    ax_signal.grid(axis="y", which="both", color="#e5e7eb", linewidth=0.6)

    x = np.arange(len(summary))
    bottom = np.zeros(len(summary))
    styles = (
        ("correct_rate", "#15803d", "Correct decisive label"),
        ("abstention_rate", "#f59e0b", "Abstention"),
        ("wrong_rate", "#b91c1c", "Wrong decisive label"),
    )
    for rate_field, color, label in styles:
        values = np.asarray([float(row[rate_field]) for row in summary])
        ax_class.bar(x, values, bottom=bottom, color=color, width=0.72, label=label)
        for xpos, low, value in zip(x, bottom, values):
            if value >= 0.035:
                ax_class.text(xpos, low + value / 2, f"{100 * value:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottom += values
    ax_class.set_xticks(x, [SOURCE_LABEL[str(row["source_model"])] for row in summary])
    ax_class.set_ylim(0, 1.08)
    ax_class.set_ylabel("Fraction of primary 10k-shot decisions")
    ax_class.set_title("b  Confirmatory result: NO-GO", loc="left", fontweight="bold")
    ax_class.text(
        0.02,
        1.015,
        "balanced accuracy 0.833 < registered 0.850",
        transform=ax_class.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#991b1b",
        fontweight="bold",
    )
    ax_class.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    ax_class.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=8)
    ax_class.text(0.98, 0.04, "0 wrong decisive labels across 960 decisions", transform=ax_class.transAxes, ha="right", va="bottom", fontsize=8.5)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_assay_schematic(output: Path) -> None:
    """Render a data-free schematic of the registered suffix intervention."""

    fig, (ax_flow, ax_curves) = plt.subplots(
        1, 2, figsize=(10.2, 3.55), gridspec_kw={"width_ratios": [1.35, 0.95]},
        constrained_layout=True,
    )
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 6)
    ax_flow.axis("off")
    ax_flow.set_title("a  Output-only paired-history intervention", loc="left", fontweight="bold")

    def box(x: float, y: float, width: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y), width, 0.75,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=color,
            edgecolor="#374151",
            linewidth=0.9,
        )
        ax_flow.add_patch(patch)
        ax_flow.text(x + width / 2, y + 0.375, text, ha="center", va="center", fontsize=9)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax_flow.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color="#4b5563", linewidth=1.1))

    ax_flow.text(0.05, 4.7, "carry", fontsize=9, fontweight="bold", color="#166534")
    box(0.65, 4.25, 1.45, "history A", "#fee2e2")
    box(0.65, 3.15, 1.45, "history B", "#dbeafe")
    arrow(2.1, 4.63, 3.0, 4.63)
    arrow(2.1, 3.53, 3.0, 3.53)
    box(3.0, 4.25, 1.55, r"state $\rho_A$", "#fee2e2")
    box(3.0, 3.15, 1.55, r"state $\rho_B$", "#dbeafe")
    arrow(4.55, 4.63, 5.45, 4.2)
    arrow(4.55, 3.53, 5.45, 4.0)
    box(5.45, 3.68, 2.05, r"same suffix $s_{1:L}$", "#dcfce7")
    arrow(7.5, 4.05, 8.25, 4.05)
    box(8.25, 3.68, 1.35, r"compare $z$", "#f3f4f6")

    ax_flow.text(0.05, 1.65, "ideal reset", fontsize=9, fontweight="bold", color="#1d4ed8")
    box(0.65, 1.2, 1.45, "history A", "#fee2e2")
    box(0.65, 0.1, 1.45, "history B", "#dbeafe")
    arrow(2.1, 1.58, 3.0, 1.03)
    arrow(2.1, 0.48, 3.0, 1.03)
    box(3.0, 0.65, 1.55, r"reset $\rho_0$", "#e0e7ff")
    arrow(4.55, 1.03, 5.45, 1.03)
    box(5.45, 0.65, 2.05, r"same suffix $s_{1:L}$", "#dcfce7")
    arrow(7.5, 1.03, 8.25, 1.03)
    box(8.25, 0.65, 1.35, r"compare $z$", "#f3f4f6")
    ax_flow.text(4.95, 5.25, "Only the output trajectories are scored; source labels are hidden.", ha="center", fontsize=8.5)

    ax_curves.set_title("b  Interpretable signatures", loc="left", fontweight="bold")
    length = np.arange(0, 13)
    recurrent = 0.65 * np.exp(-length / 11) + 0.10
    finite = np.where(length < 5, 0.68 * (1 - 0.06 * length), 0.0)
    reset = np.zeros_like(length, dtype=float)
    ax_curves.plot(length, recurrent, color="#15803d", linewidth=2.2, label="recurrent tail")
    ax_curves.step(length, finite, where="post", color="#2563eb", linewidth=2.2, label="finite cutoff $d$")
    ax_curves.plot(length, reset, color="#6b7280", linestyle="--", linewidth=1.7, label="ideal-reset reference")
    ax_curves.axhspan(0, 0.055, color="#fef3c7", alpha=0.8, label="equivalence region")
    ax_curves.set_xlabel(r"Common-suffix length $L$")
    ax_curves.set_ylabel(r"Normalized separation $R(L)$")
    ax_curves.set_ylim(-0.04, 0.9)
    ax_curves.set_xticks([0, 3, 5, 8, 12])
    ax_curves.grid(color="#e5e7eb", linewidth=0.6)
    ax_curves.legend(frameon=False, fontsize=8, loc="upper right")
    ax_curves.text(0.03, 0.05, "Low baseline signal -> abstain", transform=ax_curves.transAxes, fontsize=8.5, color="#92400e")

    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def task_source_table(
    results: Path,
    blocks: dict[str, Block],
) -> list[dict[str, object]]:
    scores = _read_csv(results / "task_link_scores.csv")
    lookup = {
        (row["physics_id"], int(row["tau"]), row["model"]): float(row["capacity_test"])
        for row in scores
    }
    rows: list[dict[str, object]] = []
    for physics_id, block in sorted(blocks.items()):
        for tau in (1, 3, 5, 8, 12):
            qrc = lookup[(physics_id, tau, "qrc")]
            qelm = lookup[(physics_id, tau, "replay_qelm_d5")]
            rows.append(
                {
                    "physics_id": physics_id,
                    "n_qubits": block.n_qubits,
                    "dissipator": block.dissipator,
                    "gamma": block.gamma,
                    "dt": block.dt,
                    "tau": tau,
                    "qrc_capacity": qrc,
                    "qelm_capacity": qelm,
                    "delta_qrc_minus_qelm": qrc - qelm,
                }
            )
    if len(rows) != 120:
        raise ValueError(f"expected 120 paired task rows, found {len(rows)}")
    return rows


def render_task_crossover(
    rows: list[dict[str, object]],
    results: Path,
    blocks: dict[str, Block],
    output: Path,
) -> None:
    study = json.loads((results / "recurrence_fingerprint_study.json").read_text(encoding="utf-8"))
    registered = study["task_link"]
    systems = _read_csv(results / "task_link_systems.csv")
    fig, (ax_tau, ax_shift) = plt.subplots(1, 2, figsize=(10.2, 4.15), constrained_layout=True)

    taus = (1, 3, 5, 8, 12)
    for dissipator in DISSIPATOR_ORDER:
        for tau in taus:
            selected = [row for row in rows if row["dissipator"] == dissipator and int(row["tau"]) == tau]
            offsets = np.linspace(-0.12, 0.12, len(selected))
            ax_tau.scatter(
                np.asarray([tau] * len(selected)) + offsets,
                [float(row["delta_qrc_minus_qelm"]) for row in selected],
                s=18,
                alpha=0.52,
                color=DISSIPATOR_COLOR[dissipator],
                edgecolor="none",
            )
    means = [np.mean([float(row["delta_qrc_minus_qelm"]) for row in rows if int(row["tau"]) == tau]) for tau in taus]
    ax_tau.plot(taus, means, color="#111827", marker="D", markersize=5, linewidth=2.0, label="Mean over 24 systems")
    ax_tau.axhline(0, color="#4b5563", linewidth=1.0)
    ax_tau.axvspan(0.5, 3.5, color="#dbeafe", alpha=0.35)
    ax_tau.axvspan(7.5, 12.5, color="#dcfce7", alpha=0.35)
    ax_tau.set_xticks(taus)
    ax_tau.set_xlabel(r"Target delay $\tau$")
    ax_tau.set_ylabel(r"Test-capacity difference $\Delta C=C_{\rm QRC}-C_{\rm QELM}$")
    ax_tau.set_title("a  Architecture ordering reverses with delay", loc="left", fontweight="bold")
    ax_tau.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    ax_tau.legend(frameon=False, loc="lower right", fontsize=8)

    for x, dissipator in enumerate(DISSIPATOR_ORDER):
        selected = [row for row in systems if blocks[row["physics_id"]].dissipator == dissipator]
        selected.sort(key=lambda row: row["physics_id"])
        offsets = np.linspace(-0.16, 0.16, len(selected))
        ax_shift.scatter(
            x + offsets,
            [float(row["crossover_shift"]) for row in selected],
            s=45,
            color=DISSIPATOR_COLOR[dissipator],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    ci_low = float(registered["mean_crossover_shift_ci_low"])
    ci_high = float(registered["mean_crossover_shift_ci_high"])
    mean_shift = float(registered["mean_crossover_shift"])
    ax_shift.axhspan(ci_low, ci_high, color="#86efac", alpha=0.28, label="Registered 95% cluster CI")
    ax_shift.axhline(mean_shift, color="#166534", linewidth=2.0, label=f"Overall mean = {mean_shift:.3f}")
    ax_shift.axhline(0, color="#4b5563", linewidth=1.0)
    ax_shift.set_xticks(range(3), [DISSIPATOR_LABEL[d] for d in DISSIPATOR_ORDER], rotation=12)
    ax_shift.set_ylabel(r"Crossover shift $\bar\Delta_{8,12}-\bar\Delta_{1,3}$")
    ax_shift.set_title("b  Held-out crossover gate passes", loc="left", fontweight="bold")
    ax_shift.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    ax_shift.legend(frameon=False, loc="upper right", fontsize=8)
    ax_shift.text(0.03, 0.04, "23/24 systems have positive shift", transform=ax_shift.transAxes, ha="left", va="bottom", fontsize=8.5)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    blocks = load_blocks(args.config)
    baselines, classification = fingerprint_source_tables(args.results, blocks)
    task_rows = task_source_table(args.results, blocks)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        args.out_dir / "fingerprint_visibility_by_system.csv",
        list(baselines[0]),
        baselines,
    )
    _write_csv(
        args.out_dir / "fingerprint_classification_summary.csv",
        list(classification[0]),
        classification,
    )
    _write_csv(
        args.out_dir / "task_crossover_by_system_tau.csv",
        list(task_rows[0]),
        task_rows,
    )
    render_fingerprint_visibility(
        baselines,
        classification,
        args.out_dir / "fingerprint_visibility.pdf",
    )
    render_assay_schematic(args.out_dir / "recurrence_assay_schematic.pdf")
    render_task_crossover(
        task_rows,
        args.results,
        blocks,
        args.out_dir / "task_crossover_heldout.pdf",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
