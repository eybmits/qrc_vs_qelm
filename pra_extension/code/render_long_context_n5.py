#!/usr/bin/env python3
"""Plot the separately frozen N5 study without selecting successful cells."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

from render_information_interface_figures import (
    COMPARISON_LAYOUT, COMPARISON_LEGEND, PALETTE,
    _base_style, _comparison_axes, _finish_comparison_axis, _metadata, _style_legend,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_long_context_n5 import validate_tables  # noqa: E402
from long_context_n5_common import read_json, sha256_file, verify_freeze  # noqa: E402

DEFAULT_RESULTS = ROOT / "results_n5_long_context_v1/run"
DEFAULT_ANALYSIS = ROOT / "results_n5_long_context_v1/analysis"
DEFAULT_OUTPUT = ROOT / "paper/figures/long_context_n5"
SERIES = (
    ("history", "old history", PALETTE["ink"], "o", "--"),
    ("qelm", "QELM, 120 features", PALETTE["onset"], "s", "-"),
    ("quadratic", "quadratic delay", PALETTE["purple"], "D", "-"),
)
FEATURE_BUDGETS = (30, 60, 120)
FEATURE_SERIES = (
    (1.0, r"$\gamma_T=1$", "s", PALETTE["onset"]),
    (4.0, r"$\gamma_T=4$", "o", "white"),
)


def _validated_suffix(results: Path, analysis: Path) -> tuple[pd.DataFrame, dict]:
    """Read the complete frozen evidence; never refit or select a best cell."""
    freeze = verify_freeze()
    summary = read_json(analysis / "long_n5_summary.json")
    manifest = read_json(analysis / "long_n5_analysis_manifest.json")
    audit = read_json(results / "long_context_run_audit.json")
    if summary["freeze"] != freeze or audit["freeze"] != freeze:
        raise ValueError("long-context figure requires the verified frozen run")
    if not summary["integrity"]["passed"]:
        raise ValueError("long-context numerical integrity failed")
    if manifest["run_audit_sha256"] != sha256_file(results / "long_context_run_audit.json"):
        raise ValueError("long-context run audit changed")
    for name, digest in manifest["artifacts"].items():
        if sha256_file(analysis / name) != digest:
            raise ValueError(f"analysis source changed: {name}")
    for name, digest in audit["science_hashes"].items():
        if sha256_file(results / name) != digest:
            raise ValueError(f"numerical source changed: {name}")
    suffix = pd.read_csv(results / "suffix_decomposition.csv")
    validate_tables(pd.read_csv(results / "stream_metrics.csv"), suffix,
                    pd.read_csv(results / "system_diagnostics.csv"))
    return suffix, freeze


def validated_source(results: Path, analysis: Path) -> pd.DataFrame:
    suffix, freeze = _validated_suffix(results, analysis)
    parts = []
    for model, metric, series in (
        ("rr_same_identity", "history_nrmse", "history"),
        ("ptd_parallel", "total_nrmse", "qelm"),
        ("poly2_delay", "total_nrmse", "quadratic"),
    ):
        selected = suffix.loc[suffix["model"].eq(model)]
        if series == "qelm":
            selected = selected.loc[selected["feature_budget"].eq(120)]
        frame = selected[["root_id", "teacher_gamma", "window", metric]].copy()
        frame = frame.rename(columns={metric: "nrmse"})
        frame["series"] = series
        frame["metric"] = metric
        parts.append(frame)
    source = pd.concat(parts, ignore_index=True)
    source["teacher_n_qubits"] = 5
    source["student_n_qubits"] = 10
    source["qelm_feature_budget"] = 120
    source["evidence_cohort"] = "prospective_long_context_n5"
    source["suffix_source_sha256"] = sha256_file(results / "suffix_decomposition.csv")
    source["freeze_sha256"] = freeze["freeze_sha256"]
    if len(source) != 360 or not source.groupby(["teacher_gamma", "window", "series"]).size().eq(12).all():
        raise ValueError("long-context plot must retain all 360 root-series observations")
    if not np.isfinite(source["nrmse"]).all() or not source["nrmse"].gt(0).all():
        raise ValueError("log display requires strictly positive finite errors")
    return source.sort_values(["teacher_gamma", "series", "window", "root_id"]).reset_index(drop=True)


def validated_feature_source(results: Path, analysis: Path) -> pd.DataFrame:
    """Retain every root and tested feature budget at the fixed longest window."""
    suffix, freeze = _validated_suffix(results, analysis)
    selected = suffix.loc[
        suffix["model"].eq("ptd_parallel") & suffix["window"].eq(10),
        ["root_id", "system_id", "teacher_n_qubits", "student_n_qubits",
         "teacher_gamma", "window", "feature_budget", "snapshot_budget",
         "total_nrmse", "history_nrmse", "representation_nrmse", "target_variance",
         "n_suffix_groups", "branches_per_group", "n_outputs"],
    ].copy().rename(columns={"representation_nrmse": "residual_emulator_nrmse"})
    selected["evidence_cohort"] = "prospective_long_context_n5"
    selected["metric"] = "total_nrmse"
    selected["suffix_source_sha256"] = sha256_file(results / "suffix_decomposition.csv")
    selected["freeze_sha256"] = freeze["freeze_sha256"]
    return selected.sort_values(["teacher_gamma", "feature_budget", "root_id"]).reset_index(drop=True)


def _validate_feature_source(source: pd.DataFrame, feature_source: pd.DataFrame) -> None:
    """Guard the descriptive slice and its link to the unchanged a/b endpoints."""
    required = {
        "root_id", "teacher_gamma", "window", "feature_budget", "total_nrmse",
        "teacher_n_qubits", "student_n_qubits",
    }
    if missing := required - set(feature_source):
        raise ValueError(f"feature source lacks columns: {sorted(missing)}")
    keys = ["teacher_gamma", "feature_budget", "root_id"]
    if len(feature_source) != 72 or feature_source.duplicated(keys).any():
        raise ValueError("feature source must contain 72 unique root-budget observations")
    if feature_source[list(required)].isna().any().any():
        raise ValueError("feature source contains missing values")
    if (not feature_source["window"].eq(10).all()
            or not feature_source["teacher_n_qubits"].eq(5).all()
            or not feature_source["student_n_qubits"].eq(10).all()):
        raise ValueError("feature source must retain d=10, a five-qubit target and ten-qubit emulator")
    expected = {(gamma, budget) for gamma in (1.0, 4.0) for budget in FEATURE_BUDGETS}
    counts = feature_source.groupby(["teacher_gamma", "feature_budget"]).size()
    if set(counts.index) != expected or not counts.eq(12).all():
        raise ValueError("feature source must retain both losses and all three tested budgets")
    errors = feature_source["total_nrmse"].to_numpy(float)
    if not np.isfinite(errors).all() or not (errors > 0).all():
        raise ValueError("feature source errors must be positive and finite")
    for (gamma, _), rows in feature_source.groupby(["teacher_gamma", "feature_budget"]):
        expected_roots = set(source.loc[source["teacher_gamma"].eq(gamma), "root_id"])
        if rows["root_id"].nunique() != 12 or set(rows["root_id"]) != expected_roots:
            raise ValueError("feature source must use the same twelve roots as panels a/b")
    endpoints = source.loc[source["series"].eq("qelm") & source["window"].eq(10),
                           ["root_id", "teacher_gamma", "nrmse"]]
    budget120 = feature_source.loc[feature_source["feature_budget"].eq(120)]
    paired = budget120.merge(endpoints, on=["root_id", "teacher_gamma"],
                             how="outer", validate="one_to_one", indicator=True)
    if (not paired["_merge"].eq("both").all()
            or not np.allclose(paired["total_nrmse"], paired["nrmse"], rtol=1e-12, atol=1e-14)):
        raise ValueError("B=120 feature errors must match the d=10 endpoints in panels a/b")


def render(source: pd.DataFrame, output: Path, *,
           feature_source: pd.DataFrame) -> tuple[Path, Path, Path, Path]:
    _validate_feature_source(source, feature_source)
    _base_style()
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(COMPARISON_LAYOUT["figure_width"], COMPARISON_LAYOUT["figure_height"]))
    axes = _comparison_axes(fig, 3)
    summary = source.groupby(["teacher_gamma", "series", "window"])["nrmse"].agg(
        median="median", lower=lambda s: s.quantile(.1), upper=lambda s: s.quantile(.9)
    )
    lower = 10.0 ** np.floor(np.log10(summary["lower"].min()))
    upper = max(3.0, 5.0 * float(summary["upper"].max()))
    for letter, gamma, ax in zip("ab", (1.0, 4.0), axes[:2], strict=True):
        _finish_comparison_axis(ax, letter)
        for series, label, color, marker, linestyle in SERIES:
            rows = summary.loc[(gamma, series)]
            median = rows["median"].to_numpy(float)
            ax.errorbar(rows.index.to_numpy(float), median,
                        yerr=np.vstack((median - rows["lower"], rows["upper"] - median)),
                        color=color, marker=marker, linestyle=linestyle, linewidth=1.1,
                        markersize=3.7, markerfacecolor="white" if series == "history" else color,
                        markeredgewidth=.6, elinewidth=.7, capsize=1.5, capthick=.7, label=label)
        ax.axhline(.1, color=PALETTE["annotation"], linestyle=(0, (4, 3)), linewidth=.85)
        ax.set_yscale("log")
        ax.set_ylim(lower / 1.2, upper)
        ax.set_xlim(1.6, 10.4)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}%"))
        ax.set_xlabel(r"window length $d$")
        # Shared description for the two identically scaled log panels frees
        # space for b's long percentage tick labels in the matched three-column grid.
        ax.set_ylabel("common-suffix error" if letter == "a" else "")
        ax.text(.95, .965, rf"$\gamma_T={gamma:g}$", transform=ax.transAxes, ha="right", va="top")
        handles = [Line2D([], [], color=color, marker=marker, linestyle=linestyle,
                          markerfacecolor="white" if series == "history" else color,
                          markersize=3.7, markeredgewidth=.6, linewidth=1.1, label=label)
                   for series, label, color, marker, linestyle in SERIES]
        if gamma == 1.0:
            legend = ax.legend(handles=handles, loc="lower left", **COMPARISON_LEGEND)
            _style_legend(legend)
    feature_summary = feature_source.groupby(["teacher_gamma", "feature_budget"])["total_nrmse"].agg(
        median="median", lower=lambda s: s.quantile(.1), upper=lambda s: s.quantile(.9),
    )
    ax = axes[2]
    _finish_comparison_axis(ax, "c")
    for offset, (gamma, label, marker, face) in zip((-.09, .09), FEATURE_SERIES, strict=True):
        rows = feature_summary.loc[gamma].reindex(FEATURE_BUDGETS)
        median = rows["median"].to_numpy(float)
        # Three discrete budget categories: dodge the two loss conditions,
        # but do not imply a measured continuous feature-response curve.
        ax.errorbar(np.arange(len(FEATURE_BUDGETS)) + offset, median,
                    yerr=np.vstack((median - rows["lower"], rows["upper"] - median)),
                    color=PALETTE["onset"], marker=marker, markerfacecolor=face,
                    linestyle="none", linewidth=1.1, markersize=4.2,
                    markeredgewidth=.8, elinewidth=.7, capsize=1.5,
                    capthick=.7, label=label)
    ax.axhline(.1, color=PALETTE["annotation"], linestyle=(0, (4, 3)), linewidth=.85)
    ax.set_yscale("linear")
    feature_upper = max(.4, np.ceil(1.05 * float(feature_summary["upper"].max()) / .1) * .1)
    ax.set_ylim(0, feature_upper)
    ax.set_yticks(np.arange(0, feature_upper + .05, .1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}%"))
    ax.set_xlim(-.45, 2.45)
    ax.set_xticks(np.arange(len(FEATURE_BUDGETS)), labels=[str(b) for b in FEATURE_BUDGETS])
    ax.set_xlabel(r"measured features $B$")
    ax.set_ylabel("QELM error")
    ax.text(.95, .965, r"$d=10$", transform=ax.transAxes, ha="right", va="top")
    handles = [
        Line2D([], [], color=PALETTE["onset"], marker=marker, linestyle="none",
               markerfacecolor=face, markersize=4.2, markeredgewidth=.8, label=label)
        for _, label, marker, face in FEATURE_SERIES
    ]
    legend = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(.98, .845),
                       **COMPARISON_LEGEND)
    _style_legend(legend)
    pdf = output / "fig04_long_context_n5.pdf"
    png = output / "fig04_long_context_n5.png"
    fig.savefig(pdf, metadata=_metadata("Five-qubit long-context follow-up"))
    fig.savefig(png, dpi=320)
    plt.close(fig)
    csv = output / "fig04_long_context_n5_source.csv"
    source.to_csv(csv, index=False)
    feature_csv = output / "fig04_long_context_n5_features_source.csv"
    feature_source.to_csv(feature_csv, index=False)
    artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (pdf, png, csv, feature_csv)}
    (output / "figure_sha256.json").write_text(json.dumps(artifacts, indent=2, sort_keys=True) + "\n")
    return pdf, png, csv, feature_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for artifact in render(validated_source(args.results, args.analysis), args.out,
                           feature_source=validated_feature_source(args.results, args.analysis)):
        print(artifact)
