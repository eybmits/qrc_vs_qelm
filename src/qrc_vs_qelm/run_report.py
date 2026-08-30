"""Build gate summaries, figures, and a paper-facing report from stage outputs."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from qrc_vs_qelm import gates, run_stage1


@dataclass(frozen=True)
class LoadedStage:
    stage: int
    directory: Path
    scores: pd.DataFrame
    manifest: dict[str, Any]
    wide: pd.DataFrame
    records: list[gates.GateRecord]
    full_records: list[gates.GateRecord]
    verdict: dict[str, Any]


@dataclass(frozen=True)
class AbsentStage:
    stage: int
    reason: str


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_figure_style()

    stages: dict[int, LoadedStage] = {}
    absent: dict[int, AbsentStage] = {}
    for stage, maybe_path in (
        (1, args.stage1),
        (2, args.stage2),
        (3, args.stage3),
    ):
        loaded, missing = _load_and_evaluate_stage(stage, maybe_path)
        if loaded is not None:
            stages[stage] = loaded
        else:
            absent[stage] = missing or AbsentStage(stage, "not provided")

    _write_gate_summary(stages, out_dir / "stage_gate_summary.csv")
    figures = _write_figures(stages, out_dir)
    _write_report(stages, absent, figures, out_dir / "REPORT.md")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, default=None)
    parser.add_argument("--stage2", type=Path, default=None)
    parser.add_argument("--stage3", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results_report"))
    return parser.parse_args(argv)


def _load_and_evaluate_stage(
    stage: int,
    stage_dir: Path | None,
) -> tuple[LoadedStage | None, AbsentStage | None]:
    if stage_dir is None:
        return None, AbsentStage(stage, "not provided")

    directory = Path(stage_dir)
    scores_path = directory / f"stage{stage}_scores.csv"
    manifest_path = directory / f"stage{stage}_manifest.json"
    missing = [str(path.name) for path in (scores_path, manifest_path) if not path.exists()]
    if missing:
        reason = f"missing {', '.join(missing)} in {directory}"
        return None, AbsentStage(stage, reason)

    scores = pd.read_csv(scores_path, float_precision="round_trip")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_boot = _bootstrap_count(manifest)

    if stage == 1:
        wide = _stage1_gate_frame(scores, manifest)
        full_records = gates.evaluate_stage1_gates(wide, n_boot=n_boot)
        records = _stage1_primary_records(wide, full_records, n_boot=n_boot)
    elif stage == 2:
        wide = _stage2_gate_frame(scores)
        records = gates.evaluate_stage2_gates(wide, n_boot=n_boot)
        full_records = records
    elif stage == 3:
        wide = _stage3_gate_frame(scores)
        records = gates.evaluate_stage3_gates(wide, n_boot=n_boot)
        full_records = records
    else:
        raise ValueError(f"unsupported stage: {stage}")

    verdict = gates.stage_verdict(records, stage=stage)
    return (
        LoadedStage(
            stage=stage,
            directory=directory,
            scores=scores,
            manifest=manifest,
            wide=wide,
            records=records,
            full_records=full_records,
            verdict=verdict,
        ),
        None,
    )


def _bootstrap_count(manifest: dict[str, Any]) -> int:
    profile = str(_manifest_payload(manifest).get("profile", "smoke")).lower()
    return 200 if profile == "smoke" else 2000


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    core = manifest.get("core")
    if isinstance(core, dict) and isinstance(core.get("payload"), dict):
        return dict(core["payload"])
    return dict(manifest)


def _stage1_gate_frame(scores: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    phase = _phase_scores(scores)
    if "confirmatory" in phase.columns:
        phase = phase[_bool_series(phase["confirmatory"])].copy()
    key_cols = _available_columns(
        phase,
        ["cell_id", "input_seed", "coupling_seed", "shot_seed", "target", "family_cell", "tau", "p"],
    )
    pivot = _capacity_pivot(phase, key_cols)
    _attach_control_capacity(pivot, phase, key_cols, "linear_delay", "capacity_linear_delay")
    _attach_control_capacity(pivot, phase, key_cols, "poly_delay", "capacity_poly_delay")
    _require_columns(pivot, ["capacity_qrc", "capacity_qelm"], "Stage-1 gate input")
    if "capacity_linear_delay" not in pivot.columns:
        pivot["capacity_linear_delay"] = np.nan
    if "capacity_poly_delay" not in pivot.columns:
        pivot["capacity_poly_delay"] = np.nan
    pivot["uninformative"] = _stage1_uninformative_flags(pivot, phase, manifest)
    pivot["delta"] = pivot["capacity_qrc"] - pivot["capacity_qelm"]
    return _stable_sort(pivot, ["tau", "p", "family_cell", "cell_id"])


def _stage2_gate_frame(scores: pd.DataFrame) -> pd.DataFrame:
    phase = _phase_scores(scores)
    if "confirmatory" in phase.columns:
        phase = phase[_bool_series(phase["confirmatory"])].copy()
    key_cols = _available_columns(
        phase,
        [
            "shots",
            "cell_id",
            "input_seed",
            "coupling_seed",
            "shot_seed",
            "target",
            "family_cell",
            "tau",
            "p",
        ],
    )
    pivot = _capacity_pivot(phase, key_cols)
    _require_columns(pivot, ["capacity_qrc", "capacity_qelm", "shots"], "Stage-2 gate input")
    pivot["delta"] = pivot["capacity_qrc"] - pivot["capacity_qelm"]
    return _stable_sort(pivot, ["shots", "tau", "p", "family_cell", "cell_id"])


def _stage3_gate_frame(scores: pd.DataFrame) -> pd.DataFrame:
    phase = _phase_scores(scores)
    if "confirmatory" in phase.columns:
        phase = phase[_bool_series(phase["confirmatory"])].copy()
    key_cols = _available_columns(
        phase,
        [
            "gamma",
            "dissipator",
            "cell_id",
            "input_seed",
            "coupling_seed",
            "shot_seed",
            "target",
            "family_cell",
            "tau",
            "p",
        ],
    )
    pivot = _capacity_pivot(phase, key_cols)
    _require_columns(pivot, ["capacity_qrc", "capacity_qelm", "gamma"], "Stage-3 gate input")

    aux_cols = [col for col in ("MC", "NC") if col in phase.columns]
    if aux_cols:
        aux = phase.groupby(key_cols, sort=False, dropna=False)[aux_cols].mean().reset_index()
        pivot = pivot.merge(aux, on=key_cols, how="left")
    if "features_dead" in phase.columns:
        dead = (
            phase.assign(features_dead=_bool_series(phase["features_dead"]))
            .groupby(key_cols, sort=False, dropna=False)["features_dead"]
            .any()
            .reset_index()
        )
        pivot = pivot.merge(dead, on=key_cols, how="left")
    pivot["delta"] = pivot["capacity_qrc"] - pivot["capacity_qelm"]
    return _stable_sort(pivot, ["dissipator", "gamma", "tau", "p", "family_cell", "cell_id"])


def _phase_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if "target_family" not in scores.columns:
        return scores.copy()
    return scores[scores["target_family"].astype(str) == "phase_diagram"].copy()


def _capacity_pivot(frame: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*key_cols, "capacity_qrc", "capacity_qelm"])
    pivot = (
        frame.groupby([*key_cols, "model"], sort=False, dropna=False)["capacity_test"]
        .first()
        .unstack("model")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rename = {
        "qrc": "capacity_qrc",
        "qelm": "capacity_qelm",
        "linear_delay": "capacity_linear_delay",
        "poly_delay": "capacity_poly_delay",
    }
    return pivot.rename(columns={key: value for key, value in rename.items() if key in pivot.columns})


def _attach_control_capacity(
    pivot: pd.DataFrame,
    long_frame: pd.DataFrame,
    key_cols: list[str],
    model: str,
    out_col: str,
) -> None:
    if out_col in pivot.columns:
        return
    if "model" in long_frame.columns:
        control = long_frame[long_frame["model"].astype(str) == model]
        if not control.empty:
            values = (
                control.groupby(key_cols, sort=False, dropna=False)["capacity_test"]
                .first()
                .reset_index(name=out_col)
            )
            merged = pivot.merge(values, on=key_cols, how="left")
            pivot[out_col] = merged[out_col].to_numpy()
            return
    if out_col in long_frame.columns:
        values = (
            long_frame.groupby(key_cols, sort=False, dropna=False)[out_col]
            .first()
            .reset_index(name=out_col)
        )
        merged = pivot.merge(values, on=key_cols, how="left")
        pivot[out_col] = merged[out_col].to_numpy()


def _stage1_uninformative_flags(
    pivot: pd.DataFrame,
    long_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> pd.Series:
    if pivot.empty:
        return pd.Series([], dtype=bool)
    threshold = _uninformative_threshold(manifest)
    if {"capacity_qrc", "capacity_qelm", "family_cell"} <= set(pivot.columns):
        cell_flags = (
            pivot.groupby("family_cell", sort=False)[["capacity_qrc", "capacity_qelm"]]
            .mean()
            .lt(threshold)
            .all(axis=1)
        )
        return pivot["family_cell"].map(cell_flags).fillna(False).astype(bool)

    if "uninformative" in long_frame.columns and "family_cell" in pivot.columns:
        flags = (
            long_frame.assign(uninformative=_bool_series(long_frame["uninformative"]))
            .groupby("family_cell", sort=False)["uninformative"]
            .any()
        )
        return pivot["family_cell"].map(flags).fillna(False).astype(bool)
    return pd.Series(False, index=pivot.index)


def _uninformative_threshold(manifest: dict[str, Any]) -> float:
    payload = _manifest_payload(manifest)
    config = payload.get("config", {})
    try:
        return float(
            config["phase_diagram"]
            .get("uninformative_exclusion", {})
            .get("both_models_test_capacity_lt", 0.05)
        )
    except Exception:
        return 0.05


def _stage1_primary_records(
    wide: pd.DataFrame,
    full_records: list[gates.GateRecord],
    *,
    n_boot: int,
) -> list[gates.GateRecord]:
    if "uninformative" not in wide.columns or not _bool_series(wide["uninformative"]).any():
        return full_records
    return run_stage1._records_with_uninformative_exclusion(  # noqa: SLF001
        full_records,
        wide,
        seed=gates.BOOTSTRAP_SEED,
        n_boot=n_boot,
    )


def _write_gate_summary(stages: dict[int, LoadedStage], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for stage in sorted(stages):
        loaded = stages[stage]
        for record in loaded.records:
            rows.append(_gate_summary_row(stage, record))
    columns = ["stage", "gate_id", "passed", "verdict", "value", "threshold", "ci", "q", "notes"]
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, lineterminator="\n")


def _gate_summary_row(stage: int, record: gates.GateRecord) -> dict[str, Any]:
    q_value = _jsonable(record.q_value)
    return {
        "stage": int(stage),
        "gate_id": record.gate_id,
        "passed": bool(record.passed),
        "verdict": record.verdict,
        "value": _json_dumps(record.value),
        "threshold": _json_dumps(record.threshold),
        "ci": _json_dumps({"low": record.ci_low, "high": record.ci_high}),
        "q": "" if q_value is None else _json_dumps(q_value),
        "notes": record.notes,
    }


def _write_figures(stages: dict[int, LoadedStage], out_dir: Path) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    if 1 in stages and not stages[1].wide.empty:
        path = out_dir / "phase_diagram.png"
        _plot_phase_diagram(stages[1], path)
        figures["phase_diagram"] = path
    if 2 in stages and not stages[2].wide.empty:
        path = out_dir / "boundary_vs_shots.png"
        _plot_boundary_vs_shots(stages[2], path)
        figures["boundary_vs_shots"] = path
    if 3 in stages and not stages[3].wide.empty:
        path = out_dir / "gamma_tradeoff.png"
        _plot_gamma_tradeoff(stages[3], path)
        figures["gamma_tradeoff"] = path
    return figures


def _plot_phase_diagram(stage1: LoadedStage, path: Path) -> None:
    cells = _phase_cell_summary(stage1)
    if cells.empty:
        return
    taus = _sorted_numeric_values(cells["tau"])
    degrees = _sorted_numeric_values(cells["p"])
    tau_index = {value: idx for idx, value in enumerate(taus)}
    degree_index = {value: idx for idx, value in enumerate(degrees)}

    matrix = np.full((len(degrees), len(taus)), np.nan, dtype=float)
    for row in cells.itertuples(index=False):
        label = str(row.label)
        if label in {"QRC", "QELM"} and not bool(row.uninformative):
            matrix[degree_index[float(row.p)], tau_index[float(row.tau)]] = float(row.delta)

    tau_edges = _cell_edges(taus)
    degree_edges = _cell_edges(degrees)
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.max(np.abs(finite))) if len(finite) else 1.0
    vmax = max(vmax, gates.TIE_BAND)
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#e5e7eb")

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    mesh = ax.pcolormesh(
        tau_edges,
        degree_edges,
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
        shading="flat",
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Delta capacity (QRC - QELM)")

    for row in cells.itertuples(index=False):
        tau = float(row.tau)
        degree = float(row.p)
        ix = tau_index[tau]
        iy = degree_index[degree]
        x0, x1 = tau_edges[ix], tau_edges[ix + 1]
        y0, y1 = degree_edges[iy], degree_edges[iy + 1]
        label = str(row.label)
        if label in {"QRC", "QELM"} and not bool(row.uninformative):
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="#111827",
                    hatch="///",
                    linewidth=1.1,
                )
            )
        if bool(row.uninformative):
            ax.scatter(tau, degree, marker="x", s=90, color="#111827", linewidths=1.8, zorder=5)

    anchors = _anchor_points(stage1.scores)
    if not anchors.empty:
        ax.scatter(
            anchors["tau"],
            anchors["p"],
            marker="o",
            s=56,
            facecolor="white",
            edgecolor="#111827",
            linewidth=1.0,
            zorder=6,
        )
        for row in anchors.itertuples(index=False):
            ax.annotate(
                str(row.label),
                (float(row.tau), float(row.p)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#111827",
                zorder=7,
            )

    x_values = list(taus) + [float(v) for v in anchors.get("tau", [])]
    y_values = list(degrees) + [float(v) for v in anchors.get("p", [])]
    _set_axis_limits(ax, x_values, y_values, tau_edges, degree_edges)
    ax.set_xticks(taus)
    ax.set_yticks(degrees)
    ax.set_xlabel("Delay tau")
    ax.set_ylabel("Degree p")
    ax.set_title(_phase_title(stage1), fontsize=10)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_boundary_vs_shots(stage2: LoadedStage, path: Path) -> None:
    counts = _shot_boundary_counts(stage2.wide)
    if counts.empty:
        return
    shot_order = _ordered_shots(counts["shots"])
    summary = (
        counts.groupby("shots", sort=False)[["qrc_count", "qelm_count"]]
        .agg(["mean", "std"])
        .reindex(shot_order)
    )
    x = np.arange(len(shot_order), dtype=float)
    qrc_mean = summary[("qrc_count", "mean")].to_numpy(dtype=float)
    qelm_mean = summary[("qelm_count", "mean")].to_numpy(dtype=float)
    qrc_std = np.nan_to_num(summary[("qrc_count", "std")].to_numpy(dtype=float), nan=0.0)
    qelm_std = np.nan_to_num(summary[("qelm_count", "std")].to_numpy(dtype=float), nan=0.0)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.errorbar(x, qrc_mean, yerr=qrc_std, marker="o", linewidth=1.8, capsize=3, label="QRC-favored")
    ax.errorbar(x, qelm_mean, yerr=qelm_std, marker="s", linewidth=1.8, capsize=3, label="QELM-favored")
    ax.set_xticks(x)
    ax.set_xticklabels([_shot_label(value) for value in shot_order])
    ax.set_xlabel("Shot tier")
    ax.set_ylabel("Favored phase cells")
    ax.set_title(f"Boundary movement under finite shots (G2a: {stage2.verdict['verdict']})")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_gamma_tradeoff(stage3: LoadedStage, path: Path) -> None:
    summary = _gamma_metric_summary(stage3.scores)
    if summary.empty:
        return
    dissipators = list(summary["dissipator"].drop_duplicates())
    n_axes = len(dissipators)
    fig, axes = plt.subplots(
        1,
        n_axes,
        figsize=(max(6.2, 5.2 * n_axes), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {"qrc": "#b91c1c", "qelm": "#1d4ed8"}
    linestyles = {"MC": "-", "NC": "--"}
    markers = {"MC": "o", "NC": "s"}
    for ax, dissipator in zip(axes[0], dissipators, strict=True):
        sub = summary[summary["dissipator"] == dissipator]
        for model in ("qrc", "qelm"):
            model_sub = sub[sub["model"] == model].sort_values("gamma")
            if model_sub.empty:
                continue
            for metric in ("MC", "NC"):
                ax.plot(
                    model_sub["gamma"],
                    model_sub[metric],
                    color=colors.get(model, "#111827"),
                    linestyle=linestyles[metric],
                    marker=markers[metric],
                    linewidth=1.6,
                    label=f"{model.upper()} {metric}",
                )
                dead = model_sub[_bool_series(model_sub["features_dead"])]
                if not dead.empty:
                    ax.scatter(
                        dead["gamma"],
                        dead[metric],
                        marker=markers[metric],
                        s=84,
                        facecolors="none",
                        edgecolors=colors.get(model, "#111827"),
                        linewidths=1.5,
                        zorder=6,
                    )
        gamma_star = _crossover_gamma(stage3.wide, str(dissipator))
        if gamma_star is not None:
            ax.axvline(gamma_star, color="#111827", linewidth=1.0, alpha=0.75)
            ax.annotate(
                f"gamma*={gamma_star:g}",
                (gamma_star, 0.98),
                xycoords=("data", "axes fraction"),
                xytext=(4, -4),
                textcoords="offset points",
                fontsize=8,
                va="top",
            )
        ax.set_title(str(dissipator))
        ax.set_xlabel("gamma")
        ax.set_ylabel("Capacity")
        ax.grid(alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(f"Gamma tradeoff by dissipator (G3a: {stage3.verdict['verdict']})", fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _phase_cell_summary(stage1: LoadedStage) -> pd.DataFrame:
    wide = stage1.wide.copy()
    wide["tau"] = pd.to_numeric(wide["tau"], errors="coerce")
    wide["p"] = pd.to_numeric(wide["p"], errors="coerce")
    labels = _g1a_label_map(stage1.full_records)
    rows: list[dict[str, Any]] = []
    for family_cell, group in wide.groupby("family_cell", sort=False):
        tau = float(group["tau"].iloc[0])
        p = float(group["p"].iloc[0])
        delta = float((group["capacity_qrc"] - group["capacity_qelm"]).mean())
        label = labels.get(str(family_cell), "tie")
        rows.append(
            {
                "family_cell": str(family_cell),
                "tau": tau,
                "p": p,
                "delta": delta,
                "label": label,
                "uninformative": bool(_bool_series(group["uninformative"]).any())
                if "uninformative" in group.columns
                else False,
            }
        )
    return _stable_sort(pd.DataFrame(rows), ["tau", "p", "family_cell"])


def _g1a_label_map(records: list[gates.GateRecord]) -> dict[str, str]:
    out: dict[str, str] = {}
    for record in records:
        if record.gate_id != "G1a":
            continue
        parsed = _parse_notes(record.notes)
        family_cell = parsed.get("family_cell")
        label = parsed.get("label", "tie")
        if family_cell is None:
            continue
        out[str(family_cell)] = label if record.passed and label in {"QRC", "QELM"} else "tie"
    return out


def _phase_title(stage1: LoadedStage) -> str:
    headline = _stage1_g1b_headline(stage1)
    with_text = _fraction_text(headline["with_value"])
    without_text = _fraction_text(headline["without_value"])
    return (
        "Phase diagram at gamma=0.3 | "
        f"G1b with exclusion {headline['with_verdict']} ({with_text}); "
        f"without exclusion {headline['without_verdict']} ({without_text})"
    )


def _stage1_g1b_headline(stage1: LoadedStage) -> dict[str, Any]:
    gate_payload = (
        _manifest_payload(stage1.manifest)
        .get("manifest_state", {})
        .get("gates", {})
    )
    with_record = _record_by_gate(stage1.records, "G1b")
    without_record = _record_by_gate(stage1.full_records, "G1b")
    manifest_with_record = _manifest_gate_record(gate_payload.get("records", []), "G1b")
    manifest_without_record = _manifest_gate_record(
        gate_payload.get("records_full_unfiltered", []),
        "G1b",
    )
    return {
        "with_value": gate_payload.get(
            "g1b_fraction_with_exclusion",
            with_record.value if with_record else {},
        ),
        "without_value": gate_payload.get(
            "g1b_fraction_without_exclusion",
            without_record.value if without_record else {},
        ),
        "with_verdict": (
            manifest_with_record.get("verdict")
            if manifest_with_record
            else (with_record.verdict if with_record else "NA")
        ),
        "without_verdict": (
            manifest_without_record.get("verdict")
            if manifest_without_record
            else (without_record.verdict if without_record else "NA")
        ),
    }


def _record_by_gate(records: list[gates.GateRecord], gate_id: str) -> gates.GateRecord | None:
    for record in records:
        if record.gate_id == gate_id:
            return record
    return None


def _manifest_gate_record(records: Any, gate_id: str) -> dict[str, Any] | None:
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, dict) and record.get("gate_id") == gate_id:
            return record
    return None


def _fraction_text(value: Any) -> str:
    if not isinstance(value, dict):
        return "fractions unavailable"
    parts = []
    for key, label in (
        ("qrc_fraction", "QRC"),
        ("qelm_fraction", "QELM"),
        ("tie_fraction", "tie"),
    ):
        if key in value:
            parts.append(f"{label}={float(value[key]):.2f}")
    if "n_cells" in value:
        parts.append(f"n={int(value['n_cells'])}")
    return ", ".join(parts) if parts else "fractions unavailable"


def _anchor_points(scores: pd.DataFrame) -> pd.DataFrame:
    if "target_family" not in scores.columns:
        return pd.DataFrame(columns=["label", "tau", "p"])
    anchors = scores[scores["target_family"].astype(str) == "anchor"].copy()
    if anchors.empty or not {"target", "tau", "p"} <= set(anchors.columns):
        return pd.DataFrame(columns=["label", "tau", "p"])
    anchors["tau"] = pd.to_numeric(anchors["tau"], errors="coerce")
    anchors["p"] = pd.to_numeric(anchors["p"], errors="coerce")
    anchors = anchors[np.isfinite(anchors["tau"]) & np.isfinite(anchors["p"])]
    rows = (
        anchors[["target", "tau", "p"]]
        .drop_duplicates()
        .rename(columns={"target": "label"})
        .sort_values(["tau", "p", "label"], kind="mergesort")
        .reset_index(drop=True)
    )
    rows["label"] = rows["label"].map(_anchor_label)
    return rows


def _anchor_label(target: Any) -> str:
    text = str(target)
    lower = text.lower()
    if lower.startswith("stm"):
        return text.upper().replace("_TAU", " tau")
    if lower.startswith("narma"):
        return text.upper()
    if "mackey" in lower:
        return "MG"
    if lower.startswith("parity"):
        return text.replace("_", " ")
    return text


def _shot_boundary_counts(wide: pd.DataFrame) -> pd.DataFrame:
    work = wide.copy()
    work["shots"] = work["shots"].map(_shot_numeric)
    work["delta"] = work["capacity_qrc"] - work["capacity_qelm"]
    strata = ["shots"]
    if "shot_seed" in work.columns:
        strata.append("shot_seed")
    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(strata, sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        shot_value = key_tuple[0]
        shot_seed = key_tuple[1] if len(key_tuple) > 1 else "all"
        cell_delta = group.groupby("family_cell", sort=False)["delta"].mean()
        rows.append(
            {
                "shots": shot_value,
                "shot_seed": shot_seed,
                "qrc_count": int((cell_delta > gates.TIE_BAND).sum()),
                "qelm_count": int((cell_delta < -gates.TIE_BAND).sum()),
                "tie_count": int((cell_delta.abs() <= gates.TIE_BAND).sum()),
            }
        )
    return pd.DataFrame(rows)


def _gamma_metric_summary(scores: pd.DataFrame) -> pd.DataFrame:
    phase = _phase_scores(scores)
    required = {"gamma", "dissipator", "model", "MC", "NC"}
    if not required <= set(phase.columns):
        return pd.DataFrame(columns=["gamma", "dissipator", "model", "MC", "NC", "features_dead"])
    id_cols = _available_columns(
        phase,
        ["gamma", "dissipator", "model", "cell_id", "input_seed", "coupling_seed", "shot_seed"],
    )
    dedup = phase.copy()
    dedup["gamma"] = pd.to_numeric(dedup["gamma"], errors="coerce")
    dedup["features_dead"] = (
        _bool_series(dedup["features_dead"]) if "features_dead" in dedup.columns else False
    )
    dedup = dedup.drop_duplicates(id_cols)
    summary = (
        dedup.groupby(["dissipator", "model", "gamma"], sort=True, dropna=False)
        .agg(MC=("MC", "mean"), NC=("NC", "mean"), features_dead=("features_dead", "any"))
        .reset_index()
    )
    return _stable_sort(summary, ["dissipator", "model", "gamma"])


def _crossover_gamma(wide: pd.DataFrame, dissipator: str | None = None) -> float | None:
    work = wide.copy()
    if dissipator is not None and "dissipator" in work.columns:
        work = work[work["dissipator"].astype(str) == str(dissipator)]
    if work.empty:
        return None
    work["gamma"] = pd.to_numeric(work["gamma"], errors="coerce")
    for gamma_value, group in work.groupby("gamma", sort=True):
        cell_means = group.groupby("family_cell", sort=False)[["capacity_qrc", "capacity_qelm"]].mean()
        fraction = float((cell_means["capacity_qelm"] >= cell_means["capacity_qrc"]).mean())
        if fraction >= 0.5:
            return float(gamma_value)
    return None


def _write_report(
    stages: dict[int, LoadedStage],
    absent: dict[int, AbsentStage],
    figures: dict[str, Path],
    path: Path,
) -> None:
    lines: list[str] = [
        "# QRC vs QELM Gate Report",
        "",
        "This report is generated only from the loaded stage score CSVs and manifests.",
        "",
        "## Inputs",
    ]
    for stage in (1, 2, 3):
        if stage in stages:
            loaded = stages[stage]
            lines.append(
                f"- Stage {stage}: `{loaded.directory.name}` ({len(loaded.scores)} score rows); "
                f"verdict `{loaded.verdict['verdict']}`."
            )
        else:
            lines.append(f"- Stage {stage}: absent ({absent[stage].reason}).")

    lines.extend(["", "## RQ1 Phase Diagram"])
    if 1 in stages:
        lines.extend(_stage1_report_lines(stages[1]))
    else:
        lines.append(f"Stage 1 is absent: {absent[1].reason}.")

    lines.extend(["", "## RQ2 Shot Scaling"])
    if 2 in stages:
        lines.extend(_stage2_report_lines(stages[2]))
    else:
        lines.append(f"Stage 2 is absent: {absent[2].reason}.")

    lines.extend(["", "## RQ3 Gamma Tradeoff"])
    if 3 in stages:
        lines.extend(_stage3_report_lines(stages[3]))
    else:
        lines.append(f"Stage 3 is absent: {absent[3].reason}.")

    lines.extend(["", "## Figures"])
    if figures:
        for name, figure_path in figures.items():
            lines.append(f"- `{figure_path.name}`: {name.replace('_', ' ')}.")
    else:
        lines.append("- No figures were produced because no usable stage data were loaded.")

    lines.extend(["", "## Limitations"])
    lines.extend(_limitations_lines(stages))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage1_report_lines(stage1: LoadedStage) -> list[str]:
    headline = _stage1_g1b_headline(stage1)
    g1c_with = _record_by_gate(stage1.records, "G1c")
    g1c_without = _record_by_gate(stage1.full_records, "G1c")
    cells = _phase_cell_summary(stage1)
    counts = cells["label"].value_counts().to_dict() if not cells.empty else {}
    uninformative = cells[cells["uninformative"]]["family_cell"].tolist() if not cells.empty else []
    lines = [
        f"Stage 1 verdict: `{stage1.verdict['verdict']}`.",
        (
            "G1b with uninformative exclusion: "
            f"`{headline['with_verdict']}` ({_fraction_text(headline['with_value'])})."
        ),
        (
            "G1b without uninformative exclusion: "
            f"`{headline['without_verdict']}` ({_fraction_text(headline['without_value'])})."
        ),
        (
            "G1c with uninformative exclusion: "
            f"`{g1c_with.verdict}` (fraction={float(g1c_with.value):.3f})."
            if g1c_with is not None
            else "G1c with uninformative exclusion: unavailable."
        ),
        (
            "G1c without uninformative exclusion: "
            f"`{g1c_without.verdict}` (fraction={float(g1c_without.value):.3f})."
            if g1c_without is not None
            else "G1c without uninformative exclusion: unavailable."
        ),
        (
            "Phase-cell labels from evaluated G1a records: "
            f"QRC={int(counts.get('QRC', 0))}, "
            f"QELM={int(counts.get('QELM', 0))}, "
            f"tie={int(counts.get('tie', 0))}."
        ),
        _region_sentence(cells, "QRC"),
        _region_sentence(cells, "QELM"),
    ]
    if uninformative:
        lines.append("Uninformative excluded cells: " + ", ".join(f"`{cell}`" for cell in uninformative) + ".")
    else:
        lines.append("No uninformative excluded cells are present in the loaded Stage-1 data.")

    anchors = _anchor_points(stage1.scores)
    if anchors.empty:
        lines.append("No anchor-task rows are present in the loaded Stage-1 scores.")
    else:
        placements = [
            f"{row.label} at tau={float(row.tau):g}, p={float(row.p):g}"
            for row in anchors.itertuples(index=False)
        ]
        lines.append("Anchor-task placements: " + "; ".join(placements) + ".")
    return lines


def _stage2_report_lines(stage2: LoadedStage) -> list[str]:
    counts = _shot_boundary_counts(stage2.wide)
    lines = [f"Stage 2 verdict: `{stage2.verdict['verdict']}`."]
    if counts.empty:
        lines.append("Shot-scaling counts could not be computed from the loaded Stage-2 data.")
        return lines
    summary = (
        counts.groupby("shots", sort=False)[["qrc_count", "qelm_count"]]
        .mean()
        .reindex(_ordered_shots(counts["shots"]))
    )
    parts = [
        f"{_shot_label(shot)}: QRC={row.qrc_count:.2f}, QELM={row.qelm_count:.2f}"
        for shot, row in summary.iterrows()
    ]
    lines.append("Mean favored-cell counts by shot tier: " + "; ".join(parts) + ".")
    return lines


def _stage3_report_lines(stage3: LoadedStage) -> list[str]:
    lines = [f"Stage 3 verdict: `{stage3.verdict['verdict']}`."]
    record = _record_by_gate(stage3.records, "G3a")
    if record is not None and isinstance(record.value, dict):
        gamma_star = record.value.get("gamma_star")
        crossover = record.value.get("crossover_fraction")
        lines.append(
            f"G3a aggregate crossover gamma*: `{gamma_star}` with max crossover fraction `{crossover}`."
        )
    for dissipator in sorted(stage3.wide["dissipator"].astype(str).dropna().unique()):
        gamma_star = _crossover_gamma(stage3.wide, dissipator)
        if gamma_star is None:
            lines.append(f"{dissipator}: no gamma where QELM reaches at least half the cells.")
        else:
            lines.append(f"{dissipator}: first gamma with QELM on at least half the cells is {gamma_star:g}.")
    collapse = _collapse_sentences(stage3.scores)
    if collapse:
        lines.extend(collapse)
    else:
        lines.append("No `features_dead=True` collapses are present in the loaded Stage-3 data.")
    return lines


def _limitations_lines(stages: dict[int, LoadedStage]) -> list[str]:
    config = _first_config(stages)
    n_qubits = _config_value(config, ["primary", "n_qubits"], _config_value(config, ["n_qubits"], "unknown"))
    degrees = _config_value(config, ["phase_diagram", "degrees"], [])
    seed_cells = _first_seed_cell_count(stages)
    degree_text = ", ".join(str(value) for value in degrees) if degrees else "unknown"
    degree5_text = (
        "degree 5 is excluded from the loaded phase-diagram gate grid"
        if 5 not in set(_as_int_list(degrees))
        else "degree 5 is present in the loaded phase-diagram gate grid"
    )
    return [
        f"- Finite scale: loaded manifests report `n_qubits={n_qubits}`; the report should not be extrapolated beyond that scale.",
        f"- Degree grid: loaded phase-diagram degrees are `{degree_text}`; {degree5_text}.",
        f"- Reproduction budget: the first loaded stage contains {seed_cells} executed seed cells in its score rows, so the evidence is moderate rather than exhaustive.",
        "- Input injection: the comparison keeps the intentional QRC/QELM input-injection asymmetry from the protocol.",
    ]


def _region_sentence(cells: pd.DataFrame, label: str) -> str:
    selected = cells[(cells["label"] == label) & ~cells["uninformative"]] if not cells.empty else cells
    if selected.empty:
        return f"{label}-favored significant regions: none in the loaded Stage-1 data."
    regions = [
        f"tau={float(row.tau):g}, p={float(row.p):g}"
        for row in selected.sort_values(["tau", "p"], kind="mergesort").itertuples(index=False)
    ]
    return f"{label}-favored significant regions: " + "; ".join(regions) + "."


def _collapse_sentences(scores: pd.DataFrame) -> list[str]:
    if "features_dead" not in scores.columns:
        return []
    phase = _phase_scores(scores).copy()
    phase["features_dead"] = _bool_series(phase["features_dead"])
    dead = phase[phase["features_dead"]]
    if dead.empty:
        return []
    rows = []
    group_cols = _available_columns(dead, ["dissipator", "model", "gamma"])
    for key, group in dead.groupby(group_cols, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = dict(zip(group_cols, key_tuple, strict=True))
        rows.append(
            "Collapse marked by `features_dead=True`: "
            f"{values.get('model', 'model')} at dissipator={values.get('dissipator', 'unknown')}, "
            f"gamma={float(values.get('gamma')):g}."
        )
    return rows


def _first_config(stages: dict[int, LoadedStage]) -> dict[str, Any]:
    for stage in sorted(stages):
        config = _manifest_payload(stages[stage].manifest).get("config")
        if isinstance(config, dict):
            return config
    return {}


def _first_seed_cell_count(stages: dict[int, LoadedStage]) -> str:
    for stage in sorted(stages):
        scores = stages[stage].scores
        seed_columns = _available_columns(
            scores,
            ["input_seed", "coupling_seed", "shot_seed"],
        )
        if seed_columns:
            return str(len(scores[seed_columns].drop_duplicates()))
        if "cell_id" in scores.columns:
            return str(scores["cell_id"].nunique(dropna=True))
    for stage in sorted(stages):
        seed_grid = _manifest_payload(stages[stage].manifest).get("seed_grid", {})
        if isinstance(seed_grid, dict):
            for key in ("n_cells", "cell_count", "seed_cells_kept"):
                if key in seed_grid:
                    return str(seed_grid[key])
    return "unknown"


def _config_value(config: dict[str, Any], keys: list[str], default: Any) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _apply_figure_style() -> None:
    for module_name, function_name in (
        ("qrc_sym.experiments.paper_experiments.figure_style", "apply_style"),
        ("qrc_sym.experiments.paper_experiments.figure_style", "set_style"),
        ("qrc_sym.experiments.paper_experiments.plot_style", "apply_style"),
    ):
        try:
            style_module = importlib.import_module(module_name)
            style_fn = getattr(style_module, function_name)
            style_fn()
            return
        except Exception:
            continue
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _parse_notes(notes: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in str(notes).split():
        if "=" not in token:
            continue
        key, value = token.split("=", maxsplit=1)
        parsed[key] = value.rstrip(";")
    return parsed


def _available_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _stable_sort(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    sort_cols = [column for column in columns if column in frame.columns]
    if not sort_cols:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _bool_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(values)
    if series.dtype == bool:
        return series.fillna(False)
    text = series.astype(str).str.lower()
    return text.isin({"true", "1", "yes", "y", "t"})


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, allow_nan=True, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sorted_numeric_values(values: pd.Series) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce")
    return [float(value) for value in sorted(numeric[np.isfinite(numeric)].unique())]


def _cell_edges(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.asarray([0.0, 1.0])
    if len(arr) == 1:
        return np.asarray([arr[0] - 0.5, arr[0] + 0.5])
    mids = (arr[:-1] + arr[1:]) / 2.0
    first = arr[0] - (mids[0] - arr[0])
    last = arr[-1] + (arr[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _set_axis_limits(
    ax: plt.Axes,
    x_values: list[float],
    y_values: list[float],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    x_min = min([float(x_edges[0]), *x_values])
    x_max = max([float(x_edges[-1]), *x_values])
    y_min = min([float(y_edges[0]), *y_values])
    y_max = max([float(y_edges[-1]), *y_values])
    x_pad = max((x_max - x_min) * 0.04, 0.25)
    y_pad = max((y_max - y_min) * 0.04, 0.25)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def _shot_numeric(value: Any) -> float:
    text = str(value).strip().lower()
    if text in {
        "inf",
        "+inf",
        "infinity",
        "+infinity",
        "∞",
        "+∞",
        "s=inf",
        "s=+inf",
        "s=infinity",
        "s=+infinity",
        "s=∞",
        "s=+∞",
    }:
        return float("inf")
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric)


def _ordered_shots(values: pd.Series) -> list[float]:
    unique = [float(value) for value in pd.Series(values).dropna().unique()]
    preferred = [float("inf"), 100000.0, 10000.0, 1000.0]
    ordered = [shot for shot in preferred if any(_shot_equal(shot, value) for value in unique)]
    ordered.extend(
        sorted(
            [value for value in unique if not any(_shot_equal(value, shot) for shot in ordered)],
            reverse=True,
        )
    )
    return ordered


def _shot_equal(left: float, right: float) -> bool:
    if math.isinf(left) and math.isinf(right):
        return True
    return bool(np.isclose(left, right))


def _shot_label(value: float) -> str:
    if math.isinf(float(value)):
        return "infinity"
    value_f = float(value)
    for numeric, label in ((100000.0, "1e5"), (10000.0, "1e4"), (1000.0, "1e3")):
        if np.isclose(value_f, numeric):
            return label
    return f"{value_f:g}"


def _as_int_list(values: Any) -> list[int]:
    try:
        return [int(value) for value in values]
    except Exception:
        return []


if __name__ == "__main__":
    raise SystemExit(main())
