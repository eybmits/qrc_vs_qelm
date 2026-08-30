#!/usr/bin/env python3
"""Render the frozen finite-window emulation decision map.

Chart contract
--------------
Question
    When can finite-window emulation work, and which resource is limiting?
Takeaway
    Dissipation or observed history moves cells out of the history-impossible
    regime; only then can feature budget separate representation-limited from
    emulated cells.

The renderer is presentation-only: it never recomputes classifications.  Each
cell retains its frozen modal class and 12-root agreement count, with sparse
hatching for non-unanimous cells and a byte-identical exported source table.
"""

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
from matplotlib.patches import Patch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "paper"
    / "figures"
    / "canonical_emulation"
    / "fig04_regime_map_source.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "figures" / "compact_pra"

CHART_QUESTION = (
    "When can finite-window emulation work, and which resource is limiting?"
)
CHART_TAKEAWAY = (
    "Dissipation or observed history moves cells out of H; only then does "
    "feature budget separate R from E."
)

ORDER = (
    "HISTORY_IMPOSSIBLE",
    "EMULATED",
    "REPRESENTATION_LIMITED",
    "UNRESOLVED",
    "DEGENERATE_TARGET",
    "MIXED",
)
COLORS = {
    "HISTORY_IMPOSSIBLE": "#7A828C",
    "EMULATED": "#2F5D8C",
    "REPRESENTATION_LIMITED": "#D88C2D",
    "UNRESOLVED": "#2A7F78",
    "DEGENERATE_TARGET": "#1F2933",
    "MIXED": "#F2F3F5",
}
CODES = {
    "HISTORY_IMPOSSIBLE": "H",
    "EMULATED": "E",
    "REPRESENTATION_LIMITED": "R",
    "UNRESOLVED": "U",
    "DEGENERATE_TARGET": "D",
    "MIXED": "mix",
}
LABELS = {
    "HISTORY_IMPOSSIBLE": "history impossible",
    "EMULATED": "emulated",
    "REPRESENTATION_LIMITED": "representation limited",
    "UNRESOLVED": "unresolved",
    "DEGENERATE_TARGET": "degenerate target",
    "MIXED": "tied mode",
}
REQUIRED_COLUMNS = {
    "teacher_n_qubits",
    "teacher_gamma",
    "window",
    "feature_budget",
    "modal_regime",
    "agreement_n",
    "n_roots",
    "regimes_present",
    "provenance_renderer_sha256",
    "provenance_config_sha256",
    "provenance_decision",
    "provenance_decision_source",
    "provenance_input_canonical_block_contrasts_csv_sha256",
    "provenance_input_canonical_gate_report_json_sha256",
    "provenance_input_canonical_manifest_json_sha256",
    "provenance_input_canonical_summary_json_sha256",
    "provenance_input_computational_resources_json_sha256",
    "provenance_input_stream_metrics_csv_sha256",
    "provenance_input_suffix_decomposition_csv_sha256",
    "provenance_input_system_diagnostics_csv_sha256",
}
PROVENANCE_SHA_COLUMNS = tuple(
    sorted(column for column in REQUIRED_COLUMNS if column.endswith("sha256"))
)
ROOT_REGIMES = frozenset(ORDER) - {"MIXED"}
EXPECTED_ROOTS = 12
EXPECTED_DECISION = "CONFIRMS_FINITE_RESOURCE_FRONTIER"
EXPECTED_DECISION_SOURCE = "canonical_gate_report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parsed_regime_counts(value: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in str(value).split(";"):
        try:
            name, count_text = item.rsplit(":", maxsplit=1)
            count = int(count_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid regimes_present entry: {value!r}") from exc
        if name not in ROOT_REGIMES or count <= 0 or name in counts:
            raise ValueError(f"invalid regimes_present entry: {value!r}")
        counts[name] = count
    if not counts:
        raise ValueError("regimes_present cannot be empty")
    return counts


def _validated_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"phase-map source lacks columns: {sorted(missing)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("phase-map source contains null required values")

    integer_columns = (
        "teacher_n_qubits",
        "window",
        "feature_budget",
        "agreement_n",
        "n_roots",
    )
    for column in integer_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if (
            not np.isfinite(numeric.to_numpy(dtype=float)).all()
            or not np.equal(numeric, np.round(numeric)).all()
        ):
            raise ValueError(f"phase-map source has non-integer {column}")
        frame[column] = numeric.astype(int)
    gamma = pd.to_numeric(frame["teacher_gamma"], errors="coerce")
    if not np.isfinite(gamma.to_numpy(dtype=float)).all():
        raise ValueError("phase-map source has invalid teacher_gamma")
    frame["teacher_gamma"] = gamma.astype(float)

    keys = ["teacher_n_qubits", "teacher_gamma", "window", "feature_budget"]
    if frame.duplicated(keys).any():
        raise ValueError("phase-map source has duplicate resource cells")
    expected = {
        (n, gamma, window, budget)
        for n in (2, 3)
        for gamma in (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)
        for window in (1, 2, 4, 6)
        for budget in (18, 36, 72)
    }
    observed = {
        (
            int(row.teacher_n_qubits),
            float(row.teacher_gamma),
            int(row.window),
            int(row.feature_budget),
        )
        for row in frame.itertuples(index=False)
    }
    if observed != expected:
        raise ValueError("phase-map source does not contain the frozen 144-cell grid")
    if not set(frame["modal_regime"]).issubset(ORDER):
        raise ValueError("phase-map source contains an unknown regime")
    if not frame["n_roots"].eq(EXPECTED_ROOTS).all():
        raise ValueError("phase-map source must contain exactly 12 roots per cell")

    for row in frame.itertuples(index=False):
        counts = _parsed_regime_counts(row.regimes_present)
        if sum(counts.values()) != int(row.n_roots):
            raise ValueError("regimes_present counts do not sum to n_roots")
        max_count = max(counts.values())
        winners = sorted(name for name, count in counts.items() if count == max_count)
        expected_modal = winners[0] if len(winners) == 1 else "MIXED"
        if int(row.agreement_n) != max_count:
            raise ValueError("agreement_n does not match regimes_present")
        if str(row.modal_regime) != expected_modal:
            raise ValueError("modal_regime does not match regimes_present")

    for column in PROVENANCE_SHA_COLUMNS:
        values = frame[column].astype(str)
        if values.nunique(dropna=False) != 1 or not values.str.fullmatch(
            r"[0-9a-f]{64}"
        ).all():
            raise ValueError(f"phase-map source has invalid {column} provenance")
    if set(frame["provenance_decision"].astype(str)) != {EXPECTED_DECISION}:
        raise ValueError("phase-map source has an unexpected frozen decision")
    if set(frame["provenance_decision_source"].astype(str)) != {
        EXPECTED_DECISION_SOURCE
    }:
        raise ValueError("phase-map source has an unexpected decision source")
    return frame.sort_values(keys).reset_index(drop=True)


def render(source_path: Path, output_dir: Path) -> dict[str, str]:
    frame = _validated_source(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.1,
            "axes.edgecolor": "#1F2933",
            "axes.linewidth": 0.8,
            "hatch.linewidth": 0.42,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    families = (2, 3)
    gammas = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)
    windows = (1, 2, 4, 6)
    budgets = (18, 36, 72)
    resource_columns = tuple(
        (gamma, window) for gamma in gammas for window in windows
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.25, 3.30),
        sharex=True,
        sharey=True,
    )

    for row, n_qubits in enumerate(families):
        ax = axes[row]
        cell = frame[frame["teacher_n_qubits"].eq(n_qubits)]
        lookup = {
            (
                float(item.teacher_gamma),
                int(item.window),
                int(item.feature_budget),
            ): item
            for item in cell.itertuples(index=False)
        }
        regimes = np.empty((len(budgets), len(resource_columns)), dtype=object)
        agreement = np.empty_like(regimes, dtype=int)
        roots = np.empty_like(regimes, dtype=int)
        for y, budget in enumerate(budgets):
            for x, (gamma, window) in enumerate(resource_columns):
                item = lookup[(gamma, window, budget)]
                regimes[y, x] = str(item.modal_regime)
                agreement[y, x] = int(item.agreement_n)
                roots[y, x] = int(item.n_roots)
        for y in range(len(budgets)):
            for x in range(len(resource_columns)):
                regime = str(regimes[y, x])
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1.0,
                        1.0,
                        facecolor=COLORS[regime],
                        edgecolor="none",
                        linewidth=0.0,
                        zorder=0,
                    )
                )
                if int(agreement[y, x]) < int(roots[y, x]):
                    dark_cell = regime in {
                        "HISTORY_IMPOSSIBLE",
                        "EMULATED",
                        "UNRESOLVED",
                        "DEGENERATE_TARGET",
                    }
                    hatch_color = (
                        (1.0, 1.0, 1.0, 0.50)
                        if dark_cell
                        else (0.10, 0.14, 0.18, 0.34)
                    )
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            hatch="//",
                            edgecolor=hatch_color,
                            linewidth=0.0,
                            zorder=1,
                        )
                    )
                text_color = (
                    "white"
                    if regime
                    in {
                        "HISTORY_IMPOSSIBLE",
                        "EMULATED",
                        "UNRESOLVED",
                        "DEGENERATE_TARGET",
                    }
                    else "#18212B"
                )
                ax.text(
                    x,
                    y,
                    CODES[regime],
                    ha="center",
                    va="center",
                    fontsize=7.6 if regime != "MIXED" else 6.2,
                    fontweight="bold",
                    color=text_color,
                    zorder=3,
                )

        ax.set_xlim(-0.5, len(resource_columns) - 0.5)
        ax.set_ylim(-0.5, len(budgets) - 0.5)
        ax.set_yticks(range(len(budgets)), budgets)
        ax.tick_params(which="both", length=0, pad=2.0)
        ax.set_xticks(np.arange(-0.5, len(resource_columns), 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, len(budgets), 1.0), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.78)
        for boundary in range(len(windows), len(resource_columns), len(windows)):
            ax.axvline(boundary - 0.5, color="white", linewidth=1.45, zorder=4)
        ax.text(
            -0.070,
            0.61,
            chr(ord("a") + row),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#18212B",
        )
        ax.text(
            -0.070,
            0.37,
            rf"$N_T={n_qubits}$",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.8,
            color="#18212B",
        )

    axes[-1].set_xticks(
        range(len(resource_columns)),
        [str(window) for _gamma in gammas for window in windows],
    )
    axes[-1].set_xlabel(
        r"within each $\gamma_T$ block: more observed history $d$ $\longrightarrow$",
        labelpad=5.0,
        color="#344454",
    )

    left, right = 0.108, 0.995
    for block, gamma in enumerate(gammas):
        center_fraction = (block * len(windows) + (len(windows) - 1) / 2 + 0.5) / len(
            resource_columns
        )
        center = left + center_fraction * (right - left)
        fig.text(
            center,
            0.845,
            rf"$\gamma_T={gamma:g}$",
            ha="center",
            va="bottom",
            fontsize=7.7,
            color="#18212B",
        )

    fig.text(
        0.55,
        0.915,
        r"teacher dissipation $\gamma_T$ increases $\longrightarrow$",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#344454",
    )
    fig.text(
        0.018,
        0.515,
        r"more student features $B$ $\longrightarrow$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.4,
        color="#344454",
    )
    active_order = [name for name in ORDER if name in set(frame["modal_regime"])]
    handles = [
        Patch(
            facecolor=COLORS[name],
            edgecolor="#18212B",
            linewidth=0.7,
            label=f"{CODES[name]}  {LABELS[name]}",
        )
        for name in active_order
    ]
    handles.append(
        Patch(
            facecolor="white",
            edgecolor="#737B84",
            linewidth=0.7,
            hatch="//",
            label="roots disagree",
        )
    )
    fig.legend(
        handles=handles,
        ncol=6,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.55, 0.012),
        fontsize=6.55,
        columnspacing=0.85,
        handlelength=1.35,
        handleheight=0.95,
    )
    fig.subplots_adjust(
        left=left,
        right=right,
        top=0.82,
        bottom=0.245,
        hspace=0.24,
    )

    pdf = output_dir / "fig02_phase_map_compact.pdf"
    png = output_dir / "fig02_phase_map_compact.png"
    source = output_dir / "fig02_phase_map_compact_source.csv"
    metadata = {
        "Title": "Finite-resource QRC-to-QELM emulation regimes",
        "Creator": "qrc_vs_qelm compact phase-map renderer",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, bbox_inches="tight", metadata=metadata)
    fig.savefig(png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    source.write_bytes(source_path.read_bytes())
    manifest = {path.name: _sha256(path) for path in (pdf, png, source)}

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = render(args.source, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
