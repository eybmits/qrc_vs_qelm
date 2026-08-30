"""Render the six preregistered canonical-emulation figures from study CSVs.

This script is presentation-only.  It consumes a completed canonical study
directory, writes auditable source tables and static PDF/PNG figures, and never
imports or calls the scientific runner.  Calibration inputs are marked visibly
as calibration-only in every export.

Chart contracts
---------------
1. History frontier: history NRMSE versus tested window with epsilon reference.
2. Error surface: median canonical-QELM NRMSE over window and feature budget.
3. Exact decomposition: mean total MSE stacked as history plus representation.
4. Regime map: modal categorical regime with root-level agreement displayed.
5. Matched controls: QELM and dimension-matched classical tanh, plus simple
   polynomial and linear delay controls, on the fixed fast-fading cells.
6. Forgetting fit: exponential-fit prediction versus minimum tested feasible
   window, preserving right-censoring and non-decaying fits explicitly.
7. Supplemental P6: positive-loss history NRMSE versus the dimensionless
   dissipation-window product, with root-level fits and family summaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import ScalarFormatter

from qrc_vs_qelm.canonical_emulation import fit_exponential_forgetting
from qrc_vs_qelm.canonical_gates import (
    load_canonical_artifacts,
    load_strict_json,
    verify_canonical_manifest,
)
from qrc_vs_qelm.canonical_study import load_canonical_profile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "canonical_emulation.yaml"
CANONICAL_INPUT_NAMES = (
    "stream_metrics.csv",
    "suffix_decomposition.csv",
    "system_diagnostics.csv",
    "canonical_summary.json",
    "computational_resources.json",
    "canonical_manifest.json",
)

INK = "#18212B"
MUTED = "#5F6B76"
GRID = "#D8DEE5"
BLUE = "#3978A8"
BLUE_DARK = "#174E78"
GOLD = "#D6A33D"
GOLD_DARK = "#8A6115"
CALIBRATION = "#9C4C3B"

GAMMA_ORDER = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)
GAMMA_COLORS = {
    0.0: "#C8DDEA",
    0.2: "#A6C9DF",
    0.5: "#7FB2D1",
    1.0: "#5798BE",
    2.0: "#3179A7",
    4.0: "#174E78",
}
GAMMA_MARKERS = {
    0.0: "o",
    0.2: "s",
    0.5: "D",
    1.0: "^",
    2.0: "v",
    4.0: "P",
}
GAMMA_LINESTYLES = {
    0.0: "-",
    0.2: "--",
    0.5: "-.",
    1.0: ":",
    2.0: (0, (5, 1.5)),
    4.0: (0, (3, 1, 1, 1)),
}

REGIME_ORDER = (
    "DEGENERATE_TARGET",
    "HISTORY_IMPOSSIBLE",
    "EMULATED",
    "REPRESENTATION_LIMITED",
    "UNRESOLVED",
)
REGIME_COLORS = {
    "DEGENERATE_TARGET": "#20262E",
    "HISTORY_IMPOSSIBLE": "#737B84",
    "EMULATED": "#3978A8",
    "REPRESENTATION_LIMITED": "#D18B2C",
    "UNRESOLVED": "#A477A8",
    "MIXED": "#F2F3F5",
}
REGIME_CODES = {
    "DEGENERATE_TARGET": "D",
    "HISTORY_IMPOSSIBLE": "H",
    "EMULATED": "E",
    "REPRESENTATION_LIMITED": "R",
    "UNRESOLVED": "U",
    "MIXED": "mix",
}

REQUIRED_SUFFIX_COLUMNS = {
    "root_id",
    "system_id",
    "teacher_n_qubits",
    "teacher_gamma",
    "window",
    "model",
    "feature_budget",
    "history_mse",
    "representation_mse",
    "total_mse",
    "target_variance",
    "identity_error",
    "history_nrmse",
    "representation_nrmse",
    "total_nrmse",
    "oracle_nrmse",
    "regime",
}


@dataclass(frozen=True)
class StudyTables:
    """Validated canonical study artifacts used by the renderer."""

    results: Path
    summary: dict[str, Any]
    stream: pd.DataFrame
    suffix: pd.DataFrame
    systems: pd.DataFrame
    canonical_manifest: dict[str, Any]
    gate_report: dict[str, Any] | None
    config_path: Path
    config_sha256: str
    input_hashes: dict[str, str]
    renderer_sha256: str
    profile: str
    decision: str
    decision_source: str
    status_label: str
    root_count: int
    tolerance_nrmse: float
    min_target_variance: float
    min_scaling_r_squared: float
    teacher_dt: float
    teacher_gammas: tuple[float, ...]
    windows: tuple[int, ...]
    fixed_fast_gamma: float
    fixed_fast_window: int
    max_snapshot_budget: int


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.4,
            "axes.labelsize": 9.0,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "text.color": INK,
            "legend.fontsize": 7.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _decision_and_status(
    summary: dict[str, Any],
    gate_report: dict[str, Any] | None,
) -> tuple[str, str, str]:
    """Return decision, provenance label, and visible status banner."""

    if not bool(summary.get("confirmatory", False)):
        return (
            "CALIBRATION_ONLY",
            "calibration_profile",
            "CALIBRATION ONLY • CONFIRMATORY SEALED",
        )
    if gate_report is None:
        return (
            "GATES_NOT_EVALUATED",
            "no_gate_report",
            "CONFIRMATORY OUTPUT • GATES NOT EVALUATED",
        )
    decision = str(gate_report.get("decision", "")).strip()
    if not decision or decision == "PENDING_GATES":
        raise ValueError("confirmatory gate report has no final decision")
    return (
        decision,
        "canonical_gate_report.json",
        f"CONFIRMATORY OUTPUT • {decision.replace('_', ' ')}",
    )


def _suffix_live_mask(frame: pd.DataFrame, min_target_variance: float) -> np.ndarray:
    """Return the registered suffix-panel target-liveness mask."""

    variance = frame["target_variance"].to_numpy(dtype=float)
    return np.isfinite(variance) & (variance >= float(min_target_variance))


def _validate_gate_report(
    results: Path,
    report: dict[str, Any],
    *,
    summary: dict[str, Any],
    actual_hashes: dict[str, str],
    config_sha256: str,
    fixed_fast_gamma: float,
    fixed_fast_window: int,
    max_snapshot_budget: int,
) -> dict[str, str]:
    """Validate decision provenance and return any additional input hashes."""

    for field in ("study", "profile", "confirmatory"):
        if report.get(field) != summary.get(field):
            raise ValueError(f"canonical gate report {field} disagrees with study summary")
    if report.get("schema_version") != 1:
        raise ValueError("canonical gate report has an unsupported schema version")
    provenance = report.get("evaluation_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("canonical gate report lacks evaluation_provenance")
    if str(provenance.get("config_sha256", "")) != config_sha256:
        raise ValueError("canonical gate report config hash disagrees with selected config")
    recorded_inputs = provenance.get("input_artifact_sha256")
    if not isinstance(recorded_inputs, dict):
        raise ValueError("canonical gate report lacks input artifact hashes")
    for name in CANONICAL_INPUT_NAMES:
        if str(recorded_inputs.get(name, "")) != actual_hashes[name]:
            raise ValueError(f"canonical gate report input hash mismatch for {name}")

    secondary = report.get("secondary_qelm_vs_best_classical")
    if not isinstance(secondary, dict):
        raise ValueError("canonical gate report lacks the registered classical comparison")
    expected_secondary = {
        "fixed_teacher_gamma": float(fixed_fast_gamma),
        "fixed_window": int(fixed_fast_window),
        "snapshot_budget": int(max_snapshot_budget),
    }
    for field, expected in expected_secondary.items():
        observed = secondary.get(field)
        if observed is None or float(observed) != float(expected):
            raise ValueError(f"canonical gate report {field} disagrees with selected config")

    additional: dict[str, str] = {}
    block = report.get("block_contrasts")
    if block is not None:
        if not isinstance(block, dict):
            raise ValueError("invalid block_contrasts provenance")
        name = str(block.get("path", ""))
        if not name or Path(name).name != name:
            raise ValueError("unsafe block_contrasts path")
        path = results / name
        expected_hash = str(block.get("sha256", ""))
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError("canonical block contrast hash mismatch")
        additional[name] = expected_hash
    return additional


def load_study_tables(
    results: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> StudyTables:
    """Load a manifest-bound canonical output directory without mutating it."""

    results = Path(results).resolve()
    config_path = Path(config_path).resolve()
    stream, suffix, systems, summary, canonical_manifest = load_canonical_artifacts(results)
    manifest_audit = verify_canonical_manifest(
        results,
        canonical_manifest,
        material_root=ROOT,
        config_path=config_path,
    )
    if not bool(manifest_audit.get("passed")):
        failed = sorted(
            name
            for name, passed in manifest_audit.get("checks", {}).items()
            if not bool(passed)
        )
        raise ValueError(f"canonical manifest verification failed: {failed}")
    if summary.get("schema_version") != 1:
        raise ValueError("canonical summary has an unsupported schema version")
    for field in ("study", "profile"):
        if str(summary.get(field)) != str(canonical_manifest.get(field)):
            raise ValueError(f"canonical summary and manifest disagree on {field}")
    profile = str(summary.get("profile", "unknown"))
    profile_config = load_canonical_profile(config_path, profile)
    for field in ("study", "profile"):
        if str(summary.get(field)) != str(profile_config.get(field)):
            raise ValueError(f"canonical summary and selected config disagree on {field}")
    if bool(summary.get("confirmatory")) != bool(profile_config.get("confirmatory")):
        raise ValueError("canonical summary confirmatory flag disagrees with selected config")

    tolerance_nrmse = float(profile_config["tolerance_nrmse"])
    min_target_variance = float(profile_config["min_target_variance"])
    min_scaling_r_squared = float(
        profile_config["decision"]["min_scaling_r_squared"]
    )
    teacher_dt = float(profile_config["teacher"]["dt"])
    teacher_gammas = tuple(float(value) for value in profile_config["teacher_gammas"])
    windows = tuple(int(value) for value in profile_config["windows"])
    fixed_fast_gamma = float(profile_config["decision"]["g5_gamma"])
    fixed_fast_window = int(max(windows))
    max_snapshot_budget = int(max(profile_config["snapshot_budgets"]))

    input_hashes = {
        name: _file_sha256(results / name) for name in CANONICAL_INPUT_NAMES
    }
    config_sha256 = _file_sha256(config_path)
    if config_sha256 != str(canonical_manifest.get("config_sha256", "")):
        raise ValueError("selected config hash disagrees with canonical manifest")

    gate_path = results / "canonical_gate_report.json"
    gate_report = load_strict_json(gate_path) if gate_path.is_file() else None
    if gate_report is not None:
        input_hashes.update(
            _validate_gate_report(
                results,
                gate_report,
                summary=summary,
                actual_hashes=input_hashes,
                config_sha256=config_sha256,
                fixed_fast_gamma=fixed_fast_gamma,
                fixed_fast_window=fixed_fast_window,
                max_snapshot_budget=max_snapshot_budget,
            )
        )
        input_hashes[gate_path.name] = _file_sha256(gate_path)

    missing = sorted(REQUIRED_SUFFIX_COLUMNS.difference(suffix.columns))
    if missing:
        raise ValueError(f"suffix_decomposition.csv missing columns: {missing}")
    if suffix.empty or systems.empty:
        raise ValueError("canonical study tables must be non-empty")
    if str(summary.get("study")) != "canonical_qrc_to_parallel_qelm_emulation":
        raise ValueError("unexpected canonical study identifier")
    configured_gammas = tuple(sorted(teacher_gammas))
    for artifact_name, frame in (
        ("system_diagnostics.csv", systems),
        ("suffix_decomposition.csv", suffix),
        ("stream_metrics.csv", stream),
    ):
        if "teacher_gamma" not in frame:
            raise ValueError(f"{artifact_name} lacks teacher_gamma")
        observed_gammas = tuple(
            sorted(float(value) for value in frame["teacher_gamma"].unique())
        )
        if observed_gammas != configured_gammas:
            raise ValueError(
                f"{artifact_name} gamma grid disagrees with selected config: "
                f"{observed_gammas} != {configured_gammas}"
            )
    style_keys = (
        tuple(GAMMA_COLORS),
        tuple(GAMMA_MARKERS),
        tuple(GAMMA_LINESTYLES),
    )
    if any(keys != GAMMA_ORDER for keys in style_keys):
        raise ValueError("frozen gamma style maps disagree with GAMMA_ORDER")
    unsupported_gammas = [
        gamma
        for gamma in teacher_gammas
        if gamma not in GAMMA_COLORS
        or gamma not in GAMMA_MARKERS
        or gamma not in GAMMA_LINESTYLES
    ]
    if unsupported_gammas:
        raise ValueError(f"no frozen figure style for teacher gammas: {unsupported_gammas}")
    configured_windows = tuple(sorted(windows))
    for artifact_name, frame in (
        ("suffix_decomposition.csv", suffix),
        ("stream_metrics.csv", stream),
    ):
        if "window" not in frame:
            raise ValueError(f"{artifact_name} lacks window")
        observed_windows = tuple(
            sorted(int(value) for value in frame["window"].unique())
        )
        if observed_windows != configured_windows:
            raise ValueError(
                f"{artifact_name} window grid disagrees with selected config: "
                f"{observed_windows} != {configured_windows}"
            )

    ptd = suffix[suffix["model"] == "ptd_parallel"].copy()
    tanh = suffix[suffix["model"] == "classical_tanh"].copy()
    if ptd.empty or tanh.empty:
        raise ValueError("canonical QELM and classical-tanh rows are both required")
    ptd_live = ptd.loc[_suffix_live_mask(ptd, min_target_variance)].copy()
    live_total_nrmse = ptd_live["total_nrmse"].to_numpy(dtype=float)
    if not np.all(np.isfinite(live_total_nrmse)):
        raise ValueError("suffix-live canonical QELM total_nrmse must be finite")
    if np.any(live_total_nrmse <= 0.0):
        raise ValueError("suffix-live canonical QELM total_nrmse must be positive")

    suffix_live = suffix.loc[
        _suffix_live_mask(suffix, min_target_variance)
    ].copy()
    identity_residual = np.abs(
        suffix_live["total_mse"].to_numpy(dtype=float)
        - suffix_live["history_mse"].to_numpy(dtype=float)
        - suffix_live["representation_mse"].to_numpy(dtype=float)
    )
    recorded_residual = np.abs(
        suffix_live["identity_error"].to_numpy(dtype=float)
    )
    if np.any(~np.isfinite(identity_residual)) or np.any(~np.isfinite(recorded_residual)):
        raise ValueError("suffix-live decomposition identity must be finite")
    max_residual = float(
        max(
            np.max(identity_residual, initial=0.0),
            np.max(recorded_residual, initial=0.0),
        )
    )
    if max_residual > 1e-10:
        raise ValueError(f"decomposition identity residual too large: {max_residual:.3e}")

    history_spread = (
        ptd_live.groupby(["root_id", "system_id", "window"], sort=True)["history_mse"]
        .agg(lambda values: float(np.max(values) - np.min(values)))
        .max()
    )
    if np.isfinite(history_spread) and float(history_spread) > 1e-12:
        raise ValueError("history floor changed across QELM feature budgets")

    root_count = int(systems["root_id"].nunique())
    if root_count < 1:
        raise ValueError("at least one root block is required")
    expected_roots = summary.get("counts", {}).get("roots")
    if expected_roots is not None and int(expected_roots) != root_count:
        raise ValueError("root count disagrees with canonical_summary.json")

    decision, decision_source, status_label = _decision_and_status(summary, gate_report)
    renderer_sha256 = _file_sha256(Path(__file__).resolve())
    return StudyTables(
        results=results,
        summary=summary,
        stream=stream,
        suffix=suffix,
        systems=systems,
        canonical_manifest=canonical_manifest,
        gate_report=gate_report,
        config_path=config_path,
        config_sha256=config_sha256,
        input_hashes=input_hashes,
        renderer_sha256=renderer_sha256,
        profile=profile,
        decision=decision,
        decision_source=decision_source,
        status_label=status_label,
        root_count=root_count,
        tolerance_nrmse=tolerance_nrmse,
        min_target_variance=min_target_variance,
        min_scaling_r_squared=min_scaling_r_squared,
        teacher_dt=teacher_dt,
        teacher_gammas=teacher_gammas,
        windows=windows,
        fixed_fast_gamma=fixed_fast_gamma,
        fixed_fast_window=fixed_fast_window,
        max_snapshot_budget=max_snapshot_budget,
    )


def _history_rows(tables: StudyTables) -> pd.DataFrame:
    history = tables.suffix[
        tables.suffix["model"] == "rr_same_identity"
    ].copy()
    key = [
        "root_id",
        "system_id",
        "teacher_n_qubits",
        "teacher_gamma",
        "window",
    ]
    columns = key + ["history_mse", "history_nrmse", "target_variance"]
    if history.empty or history.duplicated(key).any():
        raise ValueError(
            "rr_same_identity suffix rows must be unique per root/system/window"
        )
    return history[columns].sort_values(key).reset_index(drop=True)


def history_frontier_source(tables: StudyTables) -> pd.DataFrame:
    """Return suffix-live root summaries while retaining excluded-cell counts."""

    history = _history_rows(tables)
    group = ["teacher_n_qubits", "teacher_gamma", "window"]
    rows: list[dict[str, Any]] = []
    for keys, frame in history.groupby(group, sort=True):
        live = _suffix_live_mask(frame, tables.min_target_variance)
        values = frame["history_nrmse"].to_numpy(dtype=float)
        plotted = live & np.isfinite(values) & (values > 0.0)
        eligible_values = values[plotted]
        rows.append(
            {
                **dict(zip(group, keys, strict=True)),
                "median": (
                    float(np.median(eligible_values))
                    if len(eligible_values)
                    else np.nan
                ),
                "minimum": (
                    float(np.min(eligible_values))
                    if len(eligible_values)
                    else np.nan
                ),
                "maximum": (
                    float(np.max(eligible_values))
                    if len(eligible_values)
                    else np.nan
                ),
                "n_roots": int(len(frame)),
                "suffix_live_n": int(np.sum(live)),
                "suffix_dead_n": int(np.sum(~live)),
                "plotted_positive_finite_n": int(np.sum(plotted)),
                "excluded_live_n": int(np.sum(live & ~plotted)),
            }
        )
    return pd.DataFrame(rows).sort_values(group).reset_index(drop=True)


def qelm_error_surface_source(tables: StudyTables) -> pd.DataFrame:
    """Return suffix-live QELM errors for every d-by-B cell with dead counts."""

    ptd = tables.suffix[tables.suffix["model"] == "ptd_parallel"].copy()
    group = ["teacher_n_qubits", "teacher_gamma", "window", "feature_budget"]
    rows: list[dict[str, Any]] = []
    for keys, frame in ptd.groupby(group, sort=True):
        live = _suffix_live_mask(frame, tables.min_target_variance)
        values = frame["total_nrmse"].to_numpy(dtype=float)
        plotted = live & np.isfinite(values) & (values > 0.0)
        eligible_values = values[plotted]
        rows.append(
            {
                **dict(zip(group, keys, strict=True)),
                "median_total_nrmse": (
                    float(np.median(eligible_values))
                    if len(eligible_values)
                    else np.nan
                ),
                "minimum": (
                    float(np.min(eligible_values))
                    if len(eligible_values)
                    else np.nan
                ),
                "maximum": (
                    float(np.max(eligible_values))
                    if len(eligible_values)
                    else np.nan
                ),
                "n_roots": int(len(frame)),
                "suffix_live_n": int(np.sum(live)),
                "suffix_dead_n": int(np.sum(~live)),
                "plotted_positive_finite_n": int(np.sum(plotted)),
            }
        )
    return pd.DataFrame(rows).sort_values(group).reset_index(drop=True)


def decomposition_source(tables: StudyTables) -> pd.DataFrame:
    """Return the suffix-live additive root-mean decomposition at maximum B."""

    ptd = tables.suffix[tables.suffix["model"] == "ptd_parallel"].copy()
    max_budget = int(ptd["feature_budget"].max())
    ptd = ptd[ptd["feature_budget"] == max_budget]
    group = ["teacher_n_qubits", "teacher_gamma", "window", "feature_budget"]
    rows: list[dict[str, Any]] = []
    for keys, frame in ptd.groupby(group, sort=True):
        live = _suffix_live_mask(frame, tables.min_target_variance)
        components = frame[
            ["history_mse", "representation_mse", "total_mse"]
        ].to_numpy(dtype=float)
        finite_nonnegative = np.all(np.isfinite(components), axis=1) & np.all(
            components >= 0.0,
            axis=1,
        )
        plotted = live & finite_nonnegative
        eligible = frame.loc[plotted]
        rows.append(
            {
                **dict(zip(group, keys, strict=True)),
                "history_mse": (
                    float(eligible["history_mse"].mean())
                    if len(eligible)
                    else np.nan
                ),
                "representation_mse": (
                    float(eligible["representation_mse"].mean())
                    if len(eligible)
                    else np.nan
                ),
                "reported_total_mse": (
                    float(eligible["total_mse"].mean())
                    if len(eligible)
                    else np.nan
                ),
                "n_roots": int(len(frame)),
                "suffix_live_n": int(np.sum(live)),
                "suffix_dead_n": int(np.sum(~live)),
                "plotted_finite_n": int(np.sum(plotted)),
                "excluded_live_n": int(np.sum(live & ~finite_nonnegative)),
            }
        )
    source = pd.DataFrame(rows)
    source["stacked_total_mse"] = source["history_mse"] + source["representation_mse"]
    source["aggregation_identity_error"] = np.abs(
        source["reported_total_mse"] - source["stacked_total_mse"]
    )
    finite_identity = source["aggregation_identity_error"].dropna()
    if not finite_identity.empty and float(finite_identity.max()) > 1e-10:
        raise ValueError("root-mean decomposition is not additive")
    return source.sort_values(group).reset_index(drop=True)


def regime_map_source(tables: StudyTables) -> pd.DataFrame:
    """Return modal regime and agreement count for every canonical QELM cell."""

    ptd = tables.suffix[tables.suffix["model"] == "ptd_parallel"].copy()
    group = ["teacher_n_qubits", "teacher_gamma", "window", "feature_budget"]
    rows: list[dict[str, Any]] = []
    for keys, frame in ptd.groupby(group, sort=True):
        counts = frame["regime"].astype(str).value_counts().to_dict()
        unknown = sorted(set(counts).difference(REGIME_ORDER))
        if unknown:
            raise ValueError(f"unknown regime labels: {unknown}")
        best_count = max(counts.values())
        winners = [name for name in REGIME_ORDER if counts.get(name, 0) == best_count]
        modal = winners[0] if len(winners) == 1 else "MIXED"
        rows.append(
            {
                **dict(zip(group, keys, strict=True)),
                "modal_regime": modal,
                "agreement_n": int(best_count),
                "n_roots": int(len(frame)),
                "regimes_present": ";".join(
                    f"{name}:{counts[name]}" for name in REGIME_ORDER if name in counts
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(group).reset_index(drop=True)


def matched_control_source(tables: StudyTables) -> pd.DataFrame:
    """Return the secondary gate's fixed fast-fading stream-NRMSE panel.

    QELM and classical tanh use the same maximum feature budget.  The simple
    polynomial and linear delay controls have their native deterministic
    dimensions, which remain explicit in the returned table and figure labels.
    """

    stream = tables.stream
    required = {
        "root_id",
        "system_id",
        "teacher_n_qubits",
        "teacher_gamma",
        "window",
        "model",
        "snapshot_budget",
        "feature_budget",
        "nrmse",
    }
    missing = sorted(required.difference(stream.columns))
    if missing:
        raise ValueError(f"stream_metrics.csv missing columns for Figure 5: {missing}")
    fixed = stream[
        np.isclose(stream["teacher_gamma"], tables.fixed_fast_gamma)
        & (stream["window"] == tables.fixed_fast_window)
        & stream["model"].isin(
            ("ptd_parallel", "classical_tanh", "poly2_delay", "linear_delay")
        )
    ].copy()
    fixed = fixed[
        (
            fixed["model"].isin(("ptd_parallel", "classical_tanh"))
            & (fixed["snapshot_budget"] == tables.max_snapshot_budget)
        )
        | (
            fixed["model"].isin(("poly2_delay", "linear_delay"))
            & fixed["snapshot_budget"].isna()
        )
    ]
    keys = ["root_id", "system_id", "teacher_n_qubits"]
    fixed_history = _history_rows(tables)
    fixed_history = fixed_history[
        np.isclose(fixed_history["teacher_gamma"], tables.fixed_fast_gamma)
        & (fixed_history["window"] == tables.fixed_fast_window)
    ][keys + ["history_nrmse", "target_variance"]].rename(
        columns={
            "history_nrmse": "fixed_history_nrmse",
            "target_variance": "suffix_target_variance",
        }
    )
    control_keys = set(map(tuple, fixed[keys].drop_duplicates().itertuples(index=False, name=None)))
    history_keys = set(map(tuple, fixed_history[keys].itertuples(index=False, name=None)))
    if control_keys != history_keys:
        raise ValueError("fixed fast-fading controls lack a matching suffix history row")
    fixed = fixed.merge(
        fixed_history,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    fixed["suffix_target_live"] = (
        np.isfinite(fixed["suffix_target_variance"].to_numpy(dtype=float))
        & (
            fixed["suffix_target_variance"].to_numpy(dtype=float)
            >= tables.min_target_variance
        )
    )
    fixed["secondary_gate_eligible"] = (
        fixed["suffix_target_live"]
        & np.isfinite(fixed["fixed_history_nrmse"].to_numpy(dtype=float))
        & (fixed["fixed_history_nrmse"].to_numpy(dtype=float) <= tables.tolerance_nrmse)
    )
    fixed["secondary_gate_exclusion"] = np.where(
        fixed["secondary_gate_eligible"],
        "",
        np.where(
            ~fixed["suffix_target_live"],
            "SUFFIX_TARGET_NOT_LIVE",
            "HISTORY_NOT_FEASIBLE",
        ),
    )
    expected_models = {
        "ptd_parallel",
        "classical_tanh",
        "poly2_delay",
        "linear_delay",
    }
    for key, frame in fixed.groupby(keys, sort=True):
        models = set(frame["model"].astype(str))
        if models != expected_models or len(frame) != len(expected_models):
            raise ValueError(f"incomplete fixed fast-fading controls for {key}: {sorted(models)}")
    matched_feature_budgets = set(
        fixed.loc[
            fixed["model"].isin(("ptd_parallel", "classical_tanh")),
            "feature_budget",
        ].astype(int)
    )
    if len(matched_feature_budgets) != 1:
        raise ValueError("QELM and classical-tanh feature budgets are not matched")
    order = {
        "ptd_parallel": 0,
        "classical_tanh": 1,
        "poly2_delay": 2,
        "linear_delay": 3,
    }
    fixed["model_order"] = fixed["model"].map(order).astype(int)
    fixed["fixed_fast_fading_cell"] = True
    fixed["estimand"] = "held_out_stream_nrmse"
    fixed["stream_nrmse"] = fixed["nrmse"].astype(float)
    fixed["best_registered_classical"] = False
    fixed["best_classical_model"] = ""
    fixed["qelm_gain_over_best_classical"] = np.nan
    for _, frame in fixed.groupby(keys, sort=True):
        if frame["secondary_gate_eligible"].nunique() != 1:
            raise ValueError("secondary-gate feasibility changed within a control cell")
        if not bool(frame["secondary_gate_eligible"].iloc[0]):
            continue
        controls = frame[frame["model"].isin(("classical_tanh", "poly2_delay", "linear_delay"))]
        qelm = frame[frame["model"] == "ptd_parallel"]
        if len(controls) != 3 or len(qelm) != 1:
            raise ValueError("incomplete registered classical comparison")
        control_errors = {
            model: float(controls.loc[controls["model"] == model, "stream_nrmse"].iloc[0])
            for model in ("linear_delay", "poly2_delay", "classical_tanh")
        }
        winner = min(control_errors, key=control_errors.get)
        winner_index = controls.loc[controls["model"] == winner].index[0]
        classical_error = float(fixed.loc[winner_index, "stream_nrmse"])
        qelm_error = float(qelm["stream_nrmse"].iloc[0])
        if not np.isfinite(classical_error) or classical_error <= 0.0 or not np.isfinite(qelm_error):
            raise ValueError("non-finite stream NRMSE in registered classical comparison")
        group_indices = frame.index
        fixed.loc[winner_index, "best_registered_classical"] = True
        fixed.loc[group_indices, "best_classical_model"] = winner
        fixed.loc[group_indices, "qelm_gain_over_best_classical"] = 1.0 - qelm_error / classical_error
    columns = keys + [
        "teacher_gamma",
        "window",
        "model",
        "model_order",
        "snapshot_budget",
        "feature_budget",
        "stream_nrmse",
        "estimand",
        "best_registered_classical",
        "best_classical_model",
        "qelm_gain_over_best_classical",
        "fixed_fast_fading_cell",
        "fixed_history_nrmse",
        "suffix_target_variance",
        "suffix_target_live",
        "secondary_gate_eligible",
        "secondary_gate_exclusion",
    ]
    return fixed[columns].sort_values(keys + ["model_order"]).reset_index(drop=True)


def window_prediction_source(
    tables: StudyTables,
) -> pd.DataFrame:
    """Fit history decay while preserving censoring and degenerate targets."""

    tolerance = tables.tolerance_nrmse
    history = _history_rows(tables)
    group = ["root_id", "system_id", "teacher_n_qubits", "teacher_gamma"]
    rows: list[dict[str, Any]] = []
    for keys, frame in history.groupby(group, sort=True):
        frame = frame.sort_values("window")
        windows = frame["window"].to_numpy(dtype=float)
        witness = frame["history_nrmse"].to_numpy(dtype=float)
        variances = frame["target_variance"].to_numpy(dtype=float)
        registered_windows = np.asarray(sorted(tables.windows), dtype=float)
        trace_complete = bool(
            len(registered_windows) == 4
            and len(windows) == 4
            and np.array_equal(windows, registered_windows)
        )
        finite_trace = bool(np.all(np.isfinite(witness)))
        positive_trace = bool(finite_trace and np.all(witness > 0.0))
        target_live = bool(
            np.all(np.isfinite(variances))
            and np.all(variances >= tables.min_target_variance)
        )
        predicted: int | None = None
        fit_amplitude = np.nan
        fit_tau = np.nan
        fit_slope = np.nan
        fit_r_squared = np.nan
        fit_n_points = 0
        if not target_live:
            fit_status = "DEGENERATE_TARGET"
        elif not trace_complete:
            fit_status = "INCOMPLETE_REGISTERED_WINDOW_TRACE"
        elif not finite_trace:
            fit_status = "NONFINITE_HISTORY_TRACE"
        elif not positive_trace:
            fit_status = "NONPOSITIVE_HISTORY_TRACE"
        else:
            fit = fit_exponential_forgetting(
                windows,
                witness,
                min_witness=float(np.nextafter(0.0, 1.0)),
            )
            if fit.n_points != len(registered_windows):
                raise ValueError("exponential fit did not use the exact registered trace")
            predicted = fit.predicted_window(tolerance)
            fit_amplitude = fit.amplitude
            fit_tau = fit.tau
            fit_slope = fit.slope
            fit_r_squared = fit.r_squared
            fit_n_points = fit.n_points
            if predicted is None:
                fit_status = "NON_DECAYING_FIT"
            else:
                fit_status = "FIT"
        trace_eligible = bool(target_live and trace_complete and positive_trace)
        feasible = frame.loc[
            trace_eligible
            & np.isfinite(frame["history_nrmse"])
            & (frame["history_nrmse"] <= tolerance),
            "window",
        ]
        observed = None if feasible.empty else int(feasible.min())
        observed_right_censored = bool(trace_eligible and observed is None)
        predicted_finite = bool(predicted is not None)
        rows.append(
            {
                **dict(zip(group, keys, strict=True)),
                "tolerance_nrmse": float(tolerance),
                "max_tested_window": int(np.max(windows)),
                "registered_windows": ";".join(str(value) for value in tables.windows),
                "registered_window_count": int(len(tables.windows)),
                "trace_n_points": int(len(frame)),
                "trace_positive_finite_n": int(
                    np.sum(np.isfinite(witness) & (witness > 0.0))
                ),
                "registered_window_trace_complete": trace_complete,
                "suffix_target_live": target_live,
                "min_target_variance": tables.min_target_variance,
                "observed_min_tested_window": observed,
                "observed_right_censored": observed_right_censored,
                "predicted_window": predicted,
                "predicted_finite": predicted_finite,
                "fit_status": fit_status,
                "fit_amplitude": fit_amplitude,
                "fit_tau": fit_tau,
                "fit_slope": fit_slope,
                "fit_r_squared": fit_r_squared,
                "fit_n_points": fit_n_points,
            }
        )
    return pd.DataFrame(rows).sort_values(group).reset_index(drop=True)


def dissipation_window_scaling_source(
    tables: StudyTables,
) -> pd.DataFrame:
    """Return P6 point rows with per-root/family log-linear fit statistics.

    The fit follows the prospectively frozen secondary prediction exactly:
    ``log(history NRMSE)`` is regressed on ``gamma_T * dt * d`` separately for
    every root and teacher-size family, excluding the zero-loss condition.
    """

    teacher_dt = float(tables.teacher_dt)
    if not np.isfinite(teacher_dt) or teacher_dt <= 0.0:
        raise ValueError("teacher_dt must be finite and positive")
    source = _history_rows(tables)
    source = source[source["teacher_gamma"] > 0.0].copy()
    if source.empty:
        raise ValueError("P6 requires at least one positive-loss condition")
    source["suffix_target_live"] = (
        np.isfinite(source["target_variance"].to_numpy(dtype=float))
        & (
            source["target_variance"].to_numpy(dtype=float)
            >= tables.min_target_variance
        )
    )
    source["suffix_min_target_variance"] = tables.min_target_variance
    source["teacher_dt"] = teacher_dt
    source["scaling_x"] = (
        source["teacher_gamma"].to_numpy(dtype=float)
        * teacher_dt
        * source["window"].to_numpy(dtype=float)
    )

    positive_gammas = tuple(gamma for gamma in tables.teacher_gammas if gamma > 0.0)
    expected_cells = {
        (float(gamma), int(window))
        for gamma in positive_gammas
        for window in tables.windows
    }
    roots = sorted(str(value) for value in tables.systems["root_id"].unique())
    families = sorted(int(value) for value in tables.systems["teacher_n_qubits"].unique())
    fit_rows: list[dict[str, Any]] = []
    group = ["root_id", "teacher_n_qubits"]
    for root_id in roots:
        for teacher_n in families:
            frame = source[
                (source["root_id"].astype(str) == root_id)
                & (source["teacher_n_qubits"] == teacher_n)
                & source["suffix_target_live"]
            ].copy()
            observed_cells = {
                (float(row.teacher_gamma), int(row.window))
                for row in frame.itertuples(index=False)
            }
            intercept = np.nan
            slope = np.nan
            r_squared = np.nan
            fit_status = "INCOMPLETE_LIVE_GRID"
            if observed_cells == expected_cells and len(frame) == len(expected_cells):
                x = frame["scaling_x"].to_numpy(dtype=float)
                history_nrmse = frame["history_nrmse"].to_numpy(dtype=float)
                if not np.all(np.isfinite(history_nrmse)) or np.any(history_nrmse <= 0.0):
                    fit_status = "NONPOSITIVE_OR_NONFINITE_HISTORY"
                else:
                    log_y = np.log(history_nrmse)
                    design = np.column_stack([np.ones(len(x)), x])
                    coefficients, _, rank, _ = np.linalg.lstsq(design, log_y, rcond=None)
                    if int(rank) != 2:
                        fit_status = "RANK_DEFICIENT"
                    else:
                        fitted = design @ coefficients
                        ss_res = float(np.sum((log_y - fitted) ** 2))
                        ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
                        if not np.isfinite(ss_tot) or ss_tot <= 0.0 or not np.isfinite(ss_res):
                            fit_status = "UNDEFINED_VARIANCE"
                        else:
                            intercept = float(coefficients[0])
                            slope = float(coefficients[1])
                            r_squared = float(1.0 - ss_res / ss_tot)
                            fit_status = "FIT"
            fit_defined = bool(
                fit_status == "FIT"
                and np.isfinite(intercept)
                and np.isfinite(slope)
                and np.isfinite(r_squared)
            )
            passed = bool(
                fit_defined
                and slope < 0.0
                and r_squared >= tables.min_scaling_r_squared
            )
            fit_rows.append(
                {
                    "root_id": root_id,
                    "teacher_n_qubits": teacher_n,
                    "root_fit_status": fit_status,
                    "root_fit_defined": fit_defined,
                    "root_fit_intercept": intercept,
                    "root_fit_amplitude": (
                        float(np.exp(intercept)) if fit_defined else np.nan
                    ),
                    "root_fit_slope": slope,
                    "root_fit_r_squared": r_squared,
                    "root_fit_n_points": int(len(frame)),
                    "root_fit_min_r_squared": tables.min_scaling_r_squared,
                    "root_fit_passed": passed,
                }
            )
    fits = pd.DataFrame(fit_rows)
    source = source.merge(fits, on=group, how="left", validate="many_to_one")
    source["root_fit_predicted_nrmse"] = np.exp(
        source["root_fit_intercept"] + source["root_fit_slope"] * source["scaling_x"]
    )

    family = (
        fits.groupby("teacher_n_qubits", sort=True)
        .agg(
            family_median_slope=("root_fit_slope", "median"),
            family_median_r_squared=("root_fit_r_squared", "median"),
            family_successful_roots=("root_fit_passed", "sum"),
            family_n_roots=("root_id", "size"),
        )
        .reset_index()
    )
    source = source.merge(family, on="teacher_n_qubits", how="left", validate="many_to_one")
    source["prospective_secondary_prediction"] = True
    columns = [
        "root_id",
        "system_id",
        "teacher_n_qubits",
        "teacher_gamma",
        "teacher_dt",
        "window",
        "scaling_x",
        "history_nrmse",
        "target_variance",
        "suffix_target_live",
        "suffix_min_target_variance",
        "root_fit_status",
        "root_fit_defined",
        "root_fit_intercept",
        "root_fit_amplitude",
        "root_fit_slope",
        "root_fit_r_squared",
        "root_fit_n_points",
        "root_fit_min_r_squared",
        "root_fit_passed",
        "root_fit_predicted_nrmse",
        "family_median_slope",
        "family_median_r_squared",
        "family_successful_roots",
        "family_n_roots",
        "prospective_secondary_prediction",
    ]
    return source[columns].sort_values(
        ["teacher_n_qubits", "root_id", "teacher_gamma", "window"]
    ).reset_index(drop=True)


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        -0.12,
        1.08,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )


def _figure_header(
    fig: plt.Figure,
    *,
    title: str,
    subtitle: str,
    tables: StudyTables,
    source_note: str = "Source: canonical study CSV outputs.",
) -> None:
    fig.text(0.055, 0.975, title, ha="left", va="top", fontsize=14, fontweight="bold")
    fig.text(0.055, 0.938, subtitle, ha="left", va="top", fontsize=8.4, color=MUTED)
    fig.text(
        0.985,
        0.975,
        tables.status_label,
        ha="right",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=CALIBRATION if not bool(tables.summary.get("confirmatory", False)) else BLUE_DARK,
    )
    fig.text(
        0.055,
        0.018,
        f"{source_note} Profile={tables.profile}; independent root blocks n={tables.root_count}.",
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=MUTED,
    )
    if not bool(tables.summary.get("confirmatory", False)):
        fig.text(
            0.5,
            0.50,
            "CALIBRATION ONLY",
            ha="center",
            va="center",
            rotation=24,
            fontsize=45,
            fontweight="bold",
            color=CALIBRATION,
            alpha=0.035,
            zorder=-10,
        )


def _save_figure(fig: plt.Figure, pdf_path: Path, *, title: str) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Creator": "qrc_vs_qelm canonical figure renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        pdf_path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Software": "qrc_vs_qelm canonical figure renderer",
        },
    )
    plt.close(fig)


def render_history_frontier(
    source: pd.DataFrame,
    tables: StudyTables,
    output: Path,
) -> None:
    """Render figure 1: history NRMSE versus tested window."""

    tolerance = tables.tolerance_nrmse
    families = sorted(int(v) for v in source["teacher_n_qubits"].unique())
    fig, axes = plt.subplots(1, len(families), figsize=(10.6, 4.65), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    valid = source[
        np.isfinite(source["median"])
        & np.isfinite(source["minimum"])
        & np.isfinite(source["maximum"])
        & (source["median"] > 0.0)
        & (source["minimum"] > 0.0)
        & (source["maximum"] > 0.0)
    ]
    if valid.empty:
        y_min = tolerance / 10.0
        y_max = tolerance * 10.0
    else:
        positive_min = float(valid["minimum"].min())
        y_min = min(
            10.0 ** np.floor(np.log10(positive_min)),
            tolerance / 2.0,
        )
        y_max = max(1.0, float(valid["maximum"].max()) * 1.25)

    for panel, (ax, n_qubits) in enumerate(zip(axes, families, strict=True)):
        frame_n = source[source["teacher_n_qubits"] == n_qubits]
        for gamma in tables.teacher_gammas:
            frame = frame_n[
                np.isclose(frame_n["teacher_gamma"], gamma)
                & np.isfinite(frame_n["median"])
                & np.isfinite(frame_n["minimum"])
                & np.isfinite(frame_n["maximum"])
                & (frame_n["median"] > 0.0)
                & (frame_n["minimum"] > 0.0)
                & (frame_n["maximum"] > 0.0)
            ].sort_values("window")
            if frame.empty:
                continue
            x = frame["window"].to_numpy(dtype=float)
            y = frame["median"].to_numpy(dtype=float)
            low = y - frame["minimum"].to_numpy(dtype=float)
            high = frame["maximum"].to_numpy(dtype=float) - y
            color = GAMMA_COLORS[gamma]
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([low, high]),
                color=color,
                linestyle=GAMMA_LINESTYLES[gamma],
                marker=GAMMA_MARKERS[gamma],
                markeredgecolor=BLUE_DARK,
                markeredgewidth=0.55,
                markersize=4.8,
                linewidth=1.7,
                elinewidth=0.75,
                capsize=2.0,
                label=rf"$\gamma_T={gamma:g}$",
                zorder=3,
            )
        ax.axhline(tolerance, color=INK, linestyle=(0, (4, 2)), linewidth=1.1, zorder=2)
        ax.text(
            0.98,
            tolerance * 1.08,
            rf"$\epsilon={tolerance:.2f}$",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.6,
            color=INK,
        )
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(sorted(source["window"].unique()))
        ax.set_xlabel(r"Tested window $d$")
        ax.set_title(rf"Teacher family $N_T={n_qubits}$", loc="left", fontweight="bold")
        ax.grid(axis="y", which="both", color=GRID, linewidth=0.55, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        _panel_label(ax, chr(ord("a") + panel))
    axes[0].set_ylabel("History NRMSE (student-independent floor)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="lower center", bbox_to_anchor=(0.53, 0.065))
    fig.subplots_adjust(left=0.09, right=0.975, top=0.84, bottom=0.22, wspace=0.14)
    _figure_header(
        fig,
        title="History floor across tested windows",
        subtitle=(
            "Median across suffix-live root blocks; vertical bars span their observed range. "
            "Dead, nonfinite, and nonpositive cells are excluded from the log scale; the line is the registered tolerance."
        ),
        tables=tables,
    )
    _save_figure(fig, output, title="History floor across tested windows")


def _matrix_for(
    frame: pd.DataFrame,
    *,
    value: str,
    budgets: list[int],
    windows: list[int],
) -> np.ndarray:
    expected = {
        (int(budget), int(window))
        for budget in budgets
        for window in windows
    }
    observed = {
        (int(row.feature_budget), int(row.window))
        for row in frame.itertuples(index=False)
    }
    if observed != expected or len(frame) != len(expected):
        raise ValueError("incomplete d-by-B grid in canonical figure source")
    pivot = frame.pivot(index="feature_budget", columns="window", values=value)
    pivot = pivot.reindex(index=budgets, columns=windows)
    matrix = pivot.to_numpy(dtype=float)
    if np.any(np.isinf(matrix)):
        raise ValueError("infinite value in d-by-B canonical figure source")
    return matrix


def render_qelm_error_surface(
    source: pd.DataFrame,
    tables: StudyTables,
    output: Path,
) -> None:
    """Render figure 2: canonical-QELM d-by-B total-error heatmaps."""

    tolerance = tables.tolerance_nrmse
    families = sorted(int(v) for v in source["teacher_n_qubits"].unique())
    gammas = sorted(float(v) for v in source["teacher_gamma"].unique())
    windows = sorted(int(v) for v in source["window"].unique())
    budgets = sorted(int(v) for v in source["feature_budget"].unique())
    values = source["median_total_nrmse"].to_numpy(dtype=float)
    valid_values = values[np.isfinite(values) & (values > 0.0)]
    if len(valid_values):
        vmin = float(np.min(valid_values))
        vmax = float(np.max(valid_values))
        if np.isclose(vmin, vmax):
            vmin *= 0.8
            vmax *= 1.2
    else:
        vmin = tolerance / 10.0
        vmax = tolerance * 10.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps["cividis"].with_extremes(bad="#ECEFF2")
    fig, axes = plt.subplots(
        len(families),
        len(gammas),
        figsize=(13.7, 5.85),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for row, n_qubits in enumerate(families):
        for col, gamma in enumerate(gammas):
            ax = axes[row, col]
            frame = source[
                (source["teacher_n_qubits"] == n_qubits)
                & np.isclose(source["teacher_gamma"], gamma)
            ]
            matrix = _matrix_for(frame, value="median_total_nrmse", budgets=budgets, windows=windows)
            image = ax.imshow(
                np.ma.masked_invalid(matrix),
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
            )
            for y, budget in enumerate(budgets):
                for x, window in enumerate(windows):
                    value = float(matrix[y, x])
                    if not np.isfinite(value) or value <= 0.0:
                        ax.text(
                            x,
                            y,
                            "D",
                            ha="center",
                            va="center",
                            fontsize=7.0,
                            fontweight="bold",
                            color=MUTED,
                        )
                        continue
                    if value <= tolerance:
                        ax.add_patch(
                            Rectangle(
                                (x - 0.47, y - 0.47),
                                0.94,
                                0.94,
                                fill=False,
                                edgecolor=INK,
                                linewidth=1.35,
                            )
                        )
                    text_color = INK if norm(value) > 0.58 else "white"
                    label = f"{value:.2f}" if value >= 0.01 else f"{value:.1e}"
                    ax.text(x, y, label, ha="center", va="center", fontsize=6.6, color=text_color)
            ax.set_xticks(range(len(windows)), windows)
            ax.set_yticks(range(len(budgets)), budgets)
            if row == len(families) - 1:
                ax.set_xlabel(r"Window $d$")
            if col == 0:
                ax.set_ylabel(rf"$N_T={n_qubits}$" + "\nFeature budget B")
                _panel_label(ax, chr(ord("a") + row))
            ax.set_title(rf"$\gamma_T={gamma:g}$", pad=5.0)
            ax.tick_params(length=0)
    assert image is not None
    fig.subplots_adjust(left=0.07, right=0.89, top=0.84, bottom=0.14, wspace=0.08, hspace=0.24)
    cax = fig.add_axes([0.91, 0.20, 0.012, 0.54])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Median total NRMSE")
    fig.legend(
        [
            Patch(facecolor="none", edgecolor=INK, linewidth=1.35),
            Patch(facecolor="#ECEFF2", edgecolor=GRID),
        ],
        [
            rf"outlined cell: NRMSE $\leq \epsilon={tolerance:.2f}$",
            "D: no suffix-live roots",
        ],
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(0.975, 0.045),
    )
    _figure_header(
        fig,
        title="Canonical QELM error surface",
        subtitle=(
            "Median total NRMSE across suffix-live roots; every facet uses the same logarithmic color scale. "
            "D marks cells with no suffix-live roots."
        ),
        tables=tables,
    )
    _save_figure(fig, output, title="Canonical QELM error surface")


def render_exact_decomposition(
    source: pd.DataFrame,
    tables: StudyTables,
    output: Path,
) -> None:
    """Render figure 3: exact stacked total-MSE decomposition."""

    families = sorted(int(v) for v in source["teacher_n_qubits"].unique())
    gammas = sorted(float(v) for v in source["teacher_gamma"].unique())
    windows = sorted(int(v) for v in source["window"].unique())
    budget = int(source["feature_budget"].max())
    finite_totals = source.loc[
        np.isfinite(source["stacked_total_mse"])
        & (source["stacked_total_mse"] >= 0.0),
        "stacked_total_mse",
    ]
    y_max = max(
        tables.tolerance_nrmse**2 * 1.5,
        float(finite_totals.max()) * 1.08 if not finite_totals.empty else 0.0,
    )
    finite_identity = source["aggregation_identity_error"].dropna()
    max_identity = float(finite_identity.max()) if not finite_identity.empty else 0.0
    fig, axes = plt.subplots(
        len(families),
        len(gammas),
        figsize=(13.7, 5.85),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, n_qubits in enumerate(families):
        for col, gamma in enumerate(gammas):
            ax = axes[row, col]
            frame = source[
                (source["teacher_n_qubits"] == n_qubits)
                & np.isclose(source["teacher_gamma"], gamma)
            ].sort_values("window")
            if list(frame["window"].astype(int)) != windows:
                raise ValueError("incomplete window coverage for decomposition figure")
            x = np.arange(len(windows))
            history = frame["history_mse"].to_numpy(dtype=float)
            representation = frame["representation_mse"].to_numpy(dtype=float)
            plotted = frame["plotted_finite_n"].to_numpy(dtype=int) > 0
            history_plot = np.where(plotted, history, 0.0)
            representation_plot = np.where(plotted, representation, 0.0)
            ax.bar(
                x,
                history_plot,
                width=0.68,
                color=BLUE,
                edgecolor=BLUE_DARK,
                linewidth=0.65,
                label="History MSE",
            )
            ax.bar(
                x,
                representation_plot,
                bottom=history_plot,
                width=0.68,
                color=GOLD,
                edgecolor=GOLD_DARK,
                linewidth=0.65,
                hatch="///",
                label="Representation MSE",
            )
            for xpos, available in zip(x, plotted, strict=True):
                if not available:
                    ax.text(
                        xpos,
                        y_max * 0.04,
                        "D",
                        ha="center",
                        va="bottom",
                        fontsize=7.0,
                        fontweight="bold",
                        color=MUTED,
                    )
            ax.set_ylim(0.0, y_max)
            ax.set_xticks(x, windows)
            if row == len(families) - 1:
                ax.set_xlabel(r"Window $d$")
            if col == 0:
                ax.set_ylabel(rf"$N_T={n_qubits}$" + "\nMean squared error")
                _panel_label(ax, chr(ord("a") + row))
            ax.set_title(rf"$\gamma_T={gamma:g}$", pad=5.0)
            ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Patch(facecolor=BLUE, edgecolor=BLUE_DARK, label="History MSE"),
        Patch(facecolor=GOLD, edgecolor=GOLD_DARK, hatch="///", label="Representation MSE"),
        Patch(facecolor="#ECEFF2", edgecolor=GRID, label="D: no suffix-live roots"),
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.53, 0.053))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.16, wspace=0.10, hspace=0.25)
    _figure_header(
        fig,
        title="Exact history–representation decomposition",
        subtitle=(
            rf"Canonical QELM at B={budget}; suffix-live root means on a common linear scale (D: none live). "
            rf"Stacked total = history + representation; max aggregation residual {max_identity:.1e}."
        ),
        tables=tables,
    )
    _save_figure(fig, output, title="Exact history-representation decomposition")


def render_regime_map(
    source: pd.DataFrame,
    tables: StudyTables,
    output: Path,
) -> None:
    """Render figure 4: categorical regime map with agreement counts."""

    families = sorted(int(v) for v in source["teacher_n_qubits"].unique())
    gammas = sorted(float(v) for v in source["teacher_gamma"].unique())
    windows = sorted(int(v) for v in source["window"].unique())
    budgets = sorted(int(v) for v in source["feature_budget"].unique())
    categories = (*REGIME_ORDER, "MIXED")
    category_index = {name: index for index, name in enumerate(categories)}
    cmap = matplotlib.colors.ListedColormap([REGIME_COLORS[name] for name in categories])
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, len(categories) + 0.5), cmap.N)
    fig, axes = plt.subplots(
        len(families),
        len(gammas),
        figsize=(13.7, 5.95),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, n_qubits in enumerate(families):
        for col, gamma in enumerate(gammas):
            ax = axes[row, col]
            frame = source[
                (source["teacher_n_qubits"] == n_qubits)
                & np.isclose(source["teacher_gamma"], gamma)
            ]
            regime_pivot = frame.pivot(index="feature_budget", columns="window", values="modal_regime")
            regime_pivot = regime_pivot.reindex(index=budgets, columns=windows)
            agreement_pivot = frame.pivot(index="feature_budget", columns="window", values="agreement_n")
            agreement_pivot = agreement_pivot.reindex(index=budgets, columns=windows)
            roots_pivot = frame.pivot(index="feature_budget", columns="window", values="n_roots")
            roots_pivot = roots_pivot.reindex(index=budgets, columns=windows)
            if regime_pivot.isna().any().any():
                raise ValueError("incomplete d-by-B grid in regime map")
            matrix = np.asarray(
                [[category_index[str(value)] for value in row_values] for row_values in regime_pivot.to_numpy()],
                dtype=int,
            )
            ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
            for y in range(len(budgets)):
                for x in range(len(windows)):
                    regime = str(regime_pivot.iloc[y, x])
                    agreement = int(agreement_pivot.iloc[y, x])
                    n_roots = int(roots_pivot.iloc[y, x])
                    if agreement < n_roots:
                        ax.add_patch(
                            Rectangle(
                                (x - 0.5, y - 0.5),
                                1.0,
                                1.0,
                                fill=False,
                                hatch="////",
                                edgecolor="#FFFFFF",
                                linewidth=0.0,
                            )
                        )
                    text_color = "white" if regime in {
                        "DEGENERATE_TARGET",
                        "HISTORY_IMPOSSIBLE",
                        "EMULATED",
                    } else INK
                    ax.text(
                        x,
                        y,
                        f"{REGIME_CODES[regime]}\n{agreement}/{n_roots}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        fontweight="bold",
                        color=text_color,
                    )
            ax.set_xticks(range(len(windows)), windows)
            ax.set_yticks(range(len(budgets)), budgets)
            if row == len(families) - 1:
                ax.set_xlabel(r"Window $d$")
            if col == 0:
                ax.set_ylabel(rf"$N_T={n_qubits}$" + "\nFeature budget B")
                _panel_label(ax, chr(ord("a") + row))
            ax.set_title(rf"$\gamma_T={gamma:g}$", pad=5.0)
            ax.tick_params(length=0)
    legend_categories = (
        "HISTORY_IMPOSSIBLE",
        "EMULATED",
        "REPRESENTATION_LIMITED",
        "UNRESOLVED",
        "DEGENERATE_TARGET",
        "MIXED",
    )
    handles = [
        Patch(
            facecolor=REGIME_COLORS[name],
            edgecolor=INK,
            label=f"{REGIME_CODES[name]}  {name.replace('_', ' ').title()}",
        )
        for name in legend_categories
    ]
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.53, 0.047))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.17, wspace=0.08, hspace=0.24)
    _figure_header(
        fig,
        title="Finite-resource emulation regimes",
        subtitle=(
            "Cell color/code is the modal root-level classification; text gives modal agreement n/N. "
            "Hatching marks root disagreement; 'mix' denotes no unique mode."
        ),
        tables=tables,
    )
    _save_figure(fig, output, title="Finite-resource emulation regimes")


def render_matched_control(
    source: pd.DataFrame,
    tables: StudyTables,
    output: Path,
) -> None:
    """Render figure 5: fixed fast-fading quantum and classical controls."""

    families = sorted(int(v) for v in source["teacher_n_qubits"].unique())
    model_order = [
        "ptd_parallel",
        "classical_tanh",
        "poly2_delay",
        "linear_delay",
    ]
    model_style = {
        "ptd_parallel": {"color": BLUE, "edge": BLUE_DARK, "marker": "o", "label": "Parallel QELM"},
        "classical_tanh": {"color": "white", "edge": BLUE_DARK, "marker": "o", "label": "Classical tanh"},
        "poly2_delay": {"color": GOLD, "edge": GOLD_DARK, "marker": "D", "label": "Polynomial delay"},
        "linear_delay": {"color": "#D4D8DD", "edge": INK, "marker": "s", "label": "Linear delay"},
    }
    budgets = {
        model: int(source.loc[source["model"] == model, "feature_budget"].iloc[0])
        for model in model_order
    }
    eligible = source[source["secondary_gate_eligible"]].copy()
    positive_errors = eligible.loc[
        np.isfinite(eligible["stream_nrmse"])
        & (eligible["stream_nrmse"] > 0.0),
        "stream_nrmse",
    ]
    if positive_errors.empty:
        y_min = tables.tolerance_nrmse / 2.0
        y_max = tables.tolerance_nrmse * 2.0
    else:
        y_min = min(
            float(positive_errors.min()) * 0.70,
            tables.tolerance_nrmse * 0.70,
        )
        y_max = max(
            float(positive_errors.max()) * 1.42,
            tables.tolerance_nrmse * 1.30,
        )
    fig, axes = plt.subplots(1, len(families), figsize=(10.6, 4.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    root_ids = sorted(str(value) for value in source["root_id"].unique())
    offsets = (
        np.array([0.0])
        if len(root_ids) == 1
        else np.linspace(-0.10, 0.10, len(root_ids))
    )
    root_offsets = {
        root_id: offset
        for root_id, offset in zip(root_ids, offsets, strict=True)
    }
    for panel, (ax, n_qubits) in enumerate(zip(axes, families, strict=True)):
        all_frame_n = source[source["teacher_n_qubits"] == n_qubits]
        frame_n = eligible[eligible["teacher_n_qubits"] == n_qubits]
        for root_id, frame_root in frame_n.groupby("root_id", sort=True):
            frame_root = frame_root.set_index("model").reindex(model_order)
            if frame_root["stream_nrmse"].isna().any():
                raise ValueError("incomplete paired control trajectory")
            x = np.arange(len(model_order), dtype=float) + root_offsets[str(root_id)]
            ax.plot(
                x,
                frame_root["stream_nrmse"].to_numpy(dtype=float),
                color="#98A2AD",
                linewidth=0.75,
                alpha=0.75,
                zorder=1,
            )
        for xpos, model in enumerate(model_order):
            frame = frame_n[frame_n["model"] == model].sort_values("root_id")
            style = model_style[model]
            if frame.empty:
                continue
            x = np.asarray([xpos + root_offsets[str(root)] for root in frame["root_id"]])
            ax.scatter(
                x,
                frame["stream_nrmse"],
                s=47,
                marker=str(style["marker"]),
                facecolor=str(style["color"]),
                edgecolor=str(style["edge"]),
                linewidth=0.9,
                zorder=3,
            )
            median = float(frame["stream_nrmse"].median())
            ax.plot(
                [xpos - 0.20, xpos + 0.20],
                [median, median],
                color=INK,
                linewidth=2.0,
                zorder=4,
            )
            ax.annotate(
                f"{median:.3f}",
                xy=(xpos, median),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=INK,
            )
        eligible_cells = frame_n[["root_id", "system_id", "teacher_n_qubits"]].drop_duplicates()
        all_cells = all_frame_n[["root_id", "system_id", "teacher_n_qubits"]].drop_duplicates()
        ax.text(
            0.02,
            0.04,
            f"history-feasible gate cells: {len(eligible_cells)}/{len(all_cells)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.1,
            color=MUTED,
        )
        ax.axhline(tables.tolerance_nrmse, color=INK, linestyle=(0, (4, 2)), linewidth=1.0)
        ax.text(
            0.98,
            tables.tolerance_nrmse * 1.05,
            rf"$\epsilon={tables.tolerance_nrmse:.2f}$",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.4,
        )
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(
            range(len(model_order)),
            [f"{model_style[model]['label']}\nB={budgets[model]}" for model in model_order],
        )
        ax.set_title(rf"Teacher family $N_T={n_qubits}$", loc="left", fontweight="bold")
        ax.grid(axis="y", which="both", color=GRID, linewidth=0.55, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        _panel_label(ax, chr(ord("a") + panel))
    axes[0].set_ylabel("Held-out stream NRMSE")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=str(model_style[model]["marker"]),
            color="none",
            markerfacecolor=str(model_style[model]["color"]),
            markeredgecolor=str(model_style[model]["edge"]),
            markersize=6.5,
            label=str(model_style[model]["label"]),
        )
        for model in model_order
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], ncol=4, loc="lower center", bbox_to_anchor=(0.52, 0.052))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.83, bottom=0.24, wspace=0.16)
    _figure_header(
        fig,
        title="Fixed fast-fading quantum and classical controls",
        subtitle=(
            rf"Secondary-gate estimand: held-out stream NRMSE at $\gamma_T={tables.fixed_fast_gamma:g},"
            rf"d={tables.fixed_fast_window}$. QELM uses {tables.max_snapshot_budget} checkpoints; "
            rf"tanh is feature-count/width matched at B={budgets['classical_tanh']}; "
            "polynomial and linear controls retain native delay-feature dimensions. Points, pairs, and medians "
            "include only suffix-live cells whose history floor is at or below the registered tolerance."
        ),
        tables=tables,
        source_note="Source: canonical held-out stream metrics with the secondary gate's suffix-history feasibility mask; not evidence of quantum advantage.",
    )
    _save_figure(fig, output, title="Fixed fast-fading quantum and classical controls")


def render_window_prediction(
    source: pd.DataFrame,
    tables: StudyTables,
    output: Path,
) -> None:
    """Render figure 6 with honest observed and predicted censoring."""

    finite = source[
        source["predicted_finite"] & source["suffix_target_live"]
    ].copy()
    if not finite.empty and np.any(
        finite["predicted_window"].to_numpy(dtype=float) < 1.0
    ):
        raise ValueError("finite predicted windows must be positive integers")
    nonfinite = source[source["fit_status"] == "NON_DECAYING_FIT"].copy()
    plotted = finite.index.union(nonfinite.index)
    excluded = source.loc[~source.index.isin(plotted)].copy()
    d_max = int(source["max_tested_window"].max())
    censored_y = d_max + 1.25
    gamma_norm = Normalize(
        vmin=min(tables.teacher_gammas),
        vmax=max(tables.teacher_gammas),
    )
    gamma_cmap = matplotlib.colormaps["Blues"]
    marker_cycle = ("o", "^", "s", "D", "P", "X")
    families = sorted(int(value) for value in source["teacher_n_qubits"].unique())
    family_markers = {
        n_qubits: marker_cycle[index % len(marker_cycle)]
        for index, n_qubits in enumerate(families)
    }
    inf_offsets = (
        np.array([1.0])
        if len(families) == 1
        else np.linspace(0.82, 1.18, len(families))
    )
    family_inf_offsets = {
        n_qubits: offset
        for n_qubits, offset in zip(
            families,
            inf_offsets,
            strict=True,
        )
    }

    fig = plt.figure(figsize=(10.8, 4.8))
    grid = fig.add_gridspec(1, 2, width_ratios=(8.8, 1.0), left=0.09, right=0.90, top=0.82, bottom=0.22, wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    ax_inf = fig.add_subplot(grid[0, 1], sharey=ax)

    for n_qubits, marker in family_markers.items():
        frame = finite[finite["teacher_n_qubits"] == n_qubits]
        observed_y = frame["observed_min_tested_window"].fillna(censored_y).to_numpy(dtype=float)
        ax.scatter(
            frame["predicted_window"],
            observed_y,
            c=frame["teacher_gamma"],
            cmap=gamma_cmap,
            norm=gamma_norm,
            marker=marker,
            s=42,
            edgecolor=INK,
            linewidth=0.45,
            alpha=0.86,
            zorder=3,
        )
        censored = frame[frame["observed_right_censored"]]
        for _, row in censored.iterrows():
            ax.annotate(
                "",
                xy=(float(row["predicted_window"]), censored_y + 0.42),
                xytext=(float(row["predicted_window"]), censored_y - 0.24),
                arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 0.65},
                zorder=2,
            )

    ax.plot([1, d_max], [1, d_max], color=INK, linestyle=(0, (4, 2)), linewidth=1.0, label="prediction = observed")
    ax.axhline(d_max, color=MUTED, linewidth=0.75, linestyle=":")
    ax.set_xscale("log")
    finite_max = (
        float(finite["predicted_window"].max()) if not finite.empty else 1.0
    )
    x_max = max(10.0, finite_max * 1.35)
    ax.set_xlim(1.0, x_max)
    ax.set_ylim(max(0.5, min(tables.windows) - 0.3), censored_y + 0.65)
    y_ticks = [float(window) for window in tables.windows] + [censored_y]
    y_labels = [str(window) for window in tables.windows] + [f">{d_max}  censored"]
    ax.set_yticks(y_ticks, y_labels)
    ax.set_xlabel(r"Exponential-fit predicted window $d_{\rm pred}$")
    ax.set_ylabel("Observed minimum tested history-feasible window")
    ax.grid(which="both", color=GRID, linewidth=0.5, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    _panel_label(ax, "a")

    for n_qubits, marker in family_markers.items():
        frame = nonfinite[nonfinite["teacher_n_qubits"] == n_qubits]
        if frame.empty:
            continue
        x = np.full(len(frame), family_inf_offsets[n_qubits])
        observed_y = frame["observed_min_tested_window"].fillna(censored_y).to_numpy(dtype=float)
        ax_inf.scatter(
            x,
            observed_y,
            c=frame["teacher_gamma"],
            cmap=gamma_cmap,
            norm=gamma_norm,
            marker=marker,
            s=42,
            edgecolor=INK,
            linewidth=0.45,
            zorder=3,
        )
        for xpos, ypos in zip(x, observed_y, strict=True):
            if np.isclose(ypos, censored_y):
                ax_inf.annotate(
                    "",
                    xy=(xpos, censored_y + 0.42),
                    xytext=(xpos, censored_y - 0.24),
                    arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 0.65},
                )
    ax_inf.set_xlim(0.70, 1.30)
    ax_inf.set_xticks([1.0], [r"$\infty$"])
    ax_inf.set_xlabel("no finite\nprediction")
    ax_inf.tick_params(axis="y", labelleft=False, left=False)
    ax_inf.spines[["top", "right", "left"]].set_visible(False)
    ax_inf.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.75)
    if nonfinite.empty:
        ax_inf.text(
            0.5,
            0.50,
            "none",
            transform=ax_inf.transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            color=MUTED,
        )
    if not excluded.empty:
        status_counts = excluded["fit_status"].value_counts().sort_index()
        status_text = "\n".join(
            f"{status.replace('_', ' ').lower()}: {count}"
            for status, count in status_counts.items()
        )
        ax_inf.text(
            0.5,
            0.02,
            f"Not plotted\n{status_text}",
            transform=ax_inf.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.3,
            color=MUTED,
        )
    _panel_label(ax_inf, "b")

    handles = [
        Line2D([0], [0], marker=family_markers[n], color="none", markerfacecolor="#7FAECC", markeredgecolor=INK, markersize=6, label=rf"$N_T={n}$")
        for n in sorted(family_markers)
    ]
    handles.extend(
        [
            Line2D([0], [0], color=INK, linestyle=(0, (4, 2)), linewidth=1.0, label="prediction = observation"),
            Line2D([0], [0], marker="$\u2191$", color=INK, linestyle="none", markersize=8, label="observed right-censoring"),
        ]
    )
    fig.legend(handles=handles, ncol=min(5, len(handles)), loc="lower center", bbox_to_anchor=(0.50, 0.052))
    cax = fig.add_axes([0.925, 0.27, 0.012, 0.42])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=gamma_norm, cmap=gamma_cmap), cax=cax)
    colorbar.set_label(r"Teacher dissipation $\gamma_T$")
    colorbar.set_ticks(tables.teacher_gammas)
    _figure_header(
        fig,
        title="Empirical forgetting fit versus observed feasible window",
        subtitle=(
            f"Log-linear fits require all four registered positive finite values at "
            f"d={{{','.join(str(value) for value in tables.windows)}}}. Observed >{d_max} is right-censored; "
            "∞ denotes a non-decaying fitted slope; incomplete, invalid, or degenerate traces are excluded and counted."
        ),
        tables=tables,
        source_note="Source: student-independent history NRMSE in canonical suffix panels; empirical diagnostic only.",
    )
    _save_figure(fig, output, title="Empirical forgetting fit versus observed feasible window")


def render_dissipation_window_scaling(
    source: pd.DataFrame,
    tables: StudyTables,
    output: Path,
) -> None:
    """Render supplemental figure 7 for prospective secondary prediction P6."""

    tolerance = tables.tolerance_nrmse
    families = sorted(int(value) for value in source["teacher_n_qubits"].unique())
    root_ids = sorted(str(value) for value in source["root_id"].unique())
    if len(root_ids) <= 5:
        root_colors = {
            root_id: color
            for root_id, color in zip(
                root_ids,
                ("#A8C9DC", "#6FA3C4", "#3978A8", "#245F8A", "#174E78")[: len(root_ids)],
                strict=True,
            )
        }
        root_markers = {
            root_id: marker
            for root_id, marker in zip(
                root_ids,
                ("o", "s", "^", "D", "P")[: len(root_ids)],
                strict=True,
            )
        }
    else:
        # Confirmatory output has twelve roots: cycle shape while retaining a
        # single sequential-blue palette and direct root-fit statistics.
        blue_values = matplotlib.colormaps["Blues"](np.linspace(0.35, 0.88, len(root_ids)))
        root_colors = {root_id: matplotlib.colors.to_hex(color) for root_id, color in zip(root_ids, blue_values, strict=True)}
        marker_cycle = ("o", "s", "^", "D", "P", "v")
        root_markers = {root_id: marker_cycle[index % len(marker_cycle)] for index, root_id in enumerate(root_ids)}

    x_min = float(source["scaling_x"].min())
    x_max = float(source["scaling_x"].max())
    positive_live = source[
        source["suffix_target_live"]
        & np.isfinite(source["history_nrmse"])
        & (source["history_nrmse"] > 0.0)
    ]
    if positive_live.empty:
        y_min, y_max = 1e-4, 1.0
    else:
        y_min = float(positive_live["history_nrmse"].min()) * 0.62
        y_max = max(1.0, float(positive_live["history_nrmse"].max()) * 1.30)
    x_grid = np.linspace(x_min, x_max, 240)
    fig, axes = plt.subplots(1, len(families), figsize=(10.7, 4.85), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for panel, (ax, n_qubits) in enumerate(zip(axes, families, strict=True)):
        frame_n = source[source["teacher_n_qubits"] == n_qubits]
        live_frame_n = frame_n[
            frame_n["suffix_target_live"]
            & np.isfinite(frame_n["history_nrmse"])
            & (frame_n["history_nrmse"] > 0.0)
        ]
        root_predictions = []
        for root_id in root_ids:
            frame = live_frame_n[live_frame_n["root_id"] == root_id].sort_values("scaling_x")
            if frame.empty:
                continue
            fit_defined = bool(frame["root_fit_defined"].iloc[0])
            intercept = float(frame["root_fit_intercept"].iloc[0])
            slope = float(frame["root_fit_slope"].iloc[0])
            ax.scatter(
                frame["scaling_x"],
                frame["history_nrmse"],
                s=19,
                marker=root_markers[root_id],
                facecolor=root_colors[root_id],
                edgecolor=BLUE_DARK,
                linewidth=0.38,
                alpha=0.58,
                zorder=2,
            )
            if fit_defined:
                prediction = np.exp(intercept + slope * x_grid)
                root_predictions.append(prediction)
                ax.plot(
                    x_grid,
                    prediction,
                    color=root_colors[root_id],
                    linewidth=1.05,
                    alpha=0.84,
                    zorder=1,
                )
        if root_predictions:
            family_prediction = np.median(np.vstack(root_predictions), axis=0)
            ax.plot(x_grid, family_prediction, color=GOLD_DARK, linewidth=2.35, zorder=4)

        root_x_summary = (
            live_frame_n.groupby(["root_id", "scaling_x"], sort=True)["history_nrmse"]
            .median()
            .reset_index()
        )
        family_points = (
            root_x_summary.groupby("scaling_x", sort=True)["history_nrmse"]
            .median()
            .reset_index()
        )
        if not family_points.empty:
            ax.scatter(
                family_points["scaling_x"],
                family_points["history_nrmse"],
                marker="D",
                s=38,
                facecolor=GOLD,
                edgecolor=GOLD_DARK,
                linewidth=0.75,
                zorder=5,
            )
        pass_n = int(frame_n["family_successful_roots"].iloc[0])
        family_n = int(frame_n["family_n_roots"].iloc[0])
        median_slope = float(frame_n["family_median_slope"].iloc[0])
        median_r2 = float(frame_n["family_median_r_squared"].iloc[0])
        undefined_n = int(
            frame_n.drop_duplicates(["root_id", "teacher_n_qubits"])[
                "root_fit_defined"
            ].eq(False).sum()
        )
        slope_text = f"{median_slope:.3f}" if np.isfinite(median_slope) else "undefined"
        r2_text = f"{median_r2:.3f}" if np.isfinite(median_r2) else "undefined"
        ax.text(
            0.97,
            0.96,
            (
                f"root fits: slope<0 and R²≥{tables.min_scaling_r_squared:.2f}  {pass_n}/{family_n}\n"
                f"median slope={slope_text}; median R²={r2_text}; undefined={undefined_n}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.4,
            bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.28", "alpha": 0.92},
            zorder=6,
        )
        ax.axhline(tolerance, color=INK, linestyle=(0, (4, 2)), linewidth=1.0)
        ax.text(
            0.98,
            tolerance * 1.08,
            rf"$\epsilon={tolerance:.2f}$",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.4,
        )
        ax.set_yscale("log")
        ax.set_xlim(max(0.0, x_min - 0.12), x_max * 1.03)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(r"Dimensionless product $x=\gamma_T\,dt\,d$")
        ax.set_title(rf"Teacher family $N_T={n_qubits}$", loc="left", fontweight="bold")
        ax.grid(which="both", color=GRID, linewidth=0.52, alpha=0.78)
        ax.spines[["top", "right"]].set_visible(False)
        _panel_label(ax, chr(ord("a") + panel))
    axes[0].set_ylabel("History NRMSE (student-independent floor)")

    root_handles = [
        Line2D(
            [0],
            [0],
            marker=root_markers[root_id],
            color=root_colors[root_id],
            markerfacecolor=root_colors[root_id],
            markeredgecolor=BLUE_DARK,
            linewidth=1.0,
            markersize=5.0,
            label=root_id,
        )
        for root_id in root_ids
    ]
    summary_handle = Line2D(
        [0],
        [0],
        marker="D",
        color=GOLD_DARK,
        markerfacecolor=GOLD,
        markeredgecolor=GOLD_DARK,
        linewidth=2.2,
        markersize=5.5,
        label="family median points / median root-fit prediction",
    )
    legend_handles = [*root_handles, summary_handle]
    legend_columns = min(4, len(legend_handles))
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        ncol=legend_columns,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.048),
    )
    fig.subplots_adjust(left=0.095, right=0.975, top=0.82, bottom=0.23, wspace=0.16)
    dt = float(source["teacher_dt"].iloc[0])
    _figure_header(
        fig,
        title="Supplement: dissipation–window scaling of the history floor",
        subtitle=(
            rf"Prospective secondary prediction, not a primary gate: positive-loss grid only; fixed teacher $dt={dt:g}$. "
            rf"Suffix-live cells use variance $\geq {tables.min_target_variance:.1e}$; "
            "thin lines are per-root fits and gold is the family median summary."
        ),
        tables=tables,
        source_note="Source: student-independent history NRMSE; zero-loss cells excluded exactly as frozen for P6.",
    )
    _save_figure(fig, output, title="Supplemental dissipation-window scaling")


def _write_source_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bind_source_provenance(frame: pd.DataFrame, tables: StudyTables) -> pd.DataFrame:
    """Attach byte-level input and renderer provenance to every source row."""

    bound = frame.copy()
    bound["provenance_renderer_sha256"] = tables.renderer_sha256
    bound["provenance_config_sha256"] = tables.config_sha256
    bound["provenance_decision"] = tables.decision
    bound["provenance_decision_source"] = tables.decision_source
    for name, digest in sorted(tables.input_hashes.items()):
        safe = name.replace(".", "_").replace("-", "_")
        bound[f"provenance_input_{safe}_sha256"] = digest
    return bound


def _assert_provenance_current(tables: StudyTables) -> None:
    """Fail before writing if any load-time input or code binding has drifted."""

    if _file_sha256(tables.config_path) != tables.config_sha256:
        raise ValueError("selected config changed after canonical figures were loaded")
    if _file_sha256(Path(__file__).resolve()) != tables.renderer_sha256:
        raise ValueError("figure renderer changed after canonical figures were loaded")
    for name, expected_hash in tables.input_hashes.items():
        if Path(name).name != name:
            raise ValueError(f"unsafe canonical figure input path: {name}")
        path = tables.results / name
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"canonical figure input changed after load: {name}")
    audit = verify_canonical_manifest(
        tables.results,
        tables.canonical_manifest,
        material_root=ROOT,
        config_path=tables.config_path,
    )
    if not bool(audit.get("passed")):
        failed = sorted(
            name
            for name, passed in audit.get("checks", {}).items()
            if not bool(passed)
        )
        raise ValueError(f"canonical manifest drifted before rendering: {failed}")


def render_all(tables: StudyTables, output_dir: Path) -> dict[str, Any]:
    """Build six primary figures plus supplemental P6 and an audit manifest."""

    _assert_provenance_current(tables)
    _configure_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_sources = {
        "fig01_history_frontier": history_frontier_source(tables),
        "fig02_qelm_error_surface": qelm_error_surface_source(tables),
        "fig03_exact_mse_decomposition": decomposition_source(tables),
        "fig04_regime_map": regime_map_source(tables),
        "fig05_matched_controls": matched_control_source(tables),
        "fig06_predicted_vs_observed_window": window_prediction_source(tables),
        "fig07_dissipation_window_scaling": dissipation_window_scaling_source(tables),
    }
    sources = {
        stem: _bind_source_provenance(frame, tables)
        for stem, frame in raw_sources.items()
    }
    for stem, frame in sources.items():
        _write_source_csv(output_dir / f"{stem}_source.csv", frame)

    render_history_frontier(
        sources["fig01_history_frontier"],
        tables,
        output_dir / "fig01_history_frontier.pdf",
    )
    render_qelm_error_surface(
        sources["fig02_qelm_error_surface"],
        tables,
        output_dir / "fig02_qelm_error_surface.pdf",
    )
    render_exact_decomposition(
        sources["fig03_exact_mse_decomposition"], tables, output_dir / "fig03_exact_mse_decomposition.pdf"
    )
    render_regime_map(
        sources["fig04_regime_map"], tables, output_dir / "fig04_regime_map.pdf"
    )
    render_matched_control(
        sources["fig05_matched_controls"], tables, output_dir / "fig05_matched_controls.pdf"
    )
    render_window_prediction(
        sources["fig06_predicted_vs_observed_window"],
        tables,
        output_dir / "fig06_predicted_vs_observed_window.pdf",
    )
    render_dissipation_window_scaling(
        sources["fig07_dissipation_window_scaling"],
        tables,
        output_dir / "fig07_dissipation_window_scaling.pdf",
    )

    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "figure_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "study": tables.summary["study"],
        "profile": tables.profile,
        "confirmatory": bool(tables.summary.get("confirmatory", False)),
        "decision": tables.decision,
        "decision_source": tables.decision_source,
        "status_label": tables.status_label,
        "root_count": tables.root_count,
        "tolerance_nrmse": tables.tolerance_nrmse,
        "config_path": str(tables.config_path),
        "config_sha256": tables.config_sha256,
        "renderer_path": str(Path(__file__).resolve()),
        "renderer_sha256": tables.renderer_sha256,
        "primary_figures": 6,
        "supplemental_figures": 1,
        "input_artifacts": dict(sorted(tables.input_hashes.items())),
        "canonical_manifest_recorded_artifacts": tables.canonical_manifest.get(
            "artifacts",
            {},
        ),
        "outputs": {path.name: _file_sha256(path) for path in files},
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    output = args.out_dir if args.out_dir is not None else args.results / "figures"
    tables = load_study_tables(args.results, args.config)
    manifest = render_all(tables, output)
    print(
        json.dumps(
            {
                "output_dir": str(Path(output).resolve()),
                "profile": manifest["profile"],
                "status_label": manifest["status_label"],
                "figures": 7,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
