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
    PALETTE, _base_style, _metadata, _style_legend, _style_quant_axis,
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


def validated_source(results: Path, analysis: Path) -> pd.DataFrame:
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


def render(source: pd.DataFrame, output: Path) -> tuple[Path, Path, Path]:
    _base_style()
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.055, 2.65))
    axes = [fig.add_axes([left, .22, .28, .28 * 7.055 / 2.65]) for left in (.10, .60)]
    summary = source.groupby(["teacher_gamma", "series", "window"])["nrmse"].agg(
        median="median", lower=lambda s: s.quantile(.1), upper=lambda s: s.quantile(.9)
    )
    lower = 10.0 ** np.floor(np.log10(summary["lower"].min()))
    upper = max(3.0, 5.0 * float(summary["upper"].max()))
    for letter, gamma, ax in zip("ab", (1.0, 4.0), axes, strict=True):
        _style_quant_axis(ax)
        for series, label, color, marker, linestyle in SERIES:
            rows = summary.loc[(gamma, series)]
            median = rows["median"].to_numpy(float)
            ax.errorbar(rows.index.to_numpy(float), median,
                        yerr=np.vstack((median - rows["lower"], rows["upper"] - median)),
                        color=color, marker=marker, linestyle=linestyle, linewidth=1.1,
                        markersize=3.7, markerfacecolor="white" if series == "history" else color,
                        elinewidth=.7, capsize=1.5, capthick=.7, label=label)
        ax.axhline(.1, color=PALETTE["annotation"], linestyle=(0, (4, 3)), linewidth=.8)
        ax.set_yscale("log")
        ax.set_ylim(lower / 1.2, upper)
        ax.set_xlim(1.6, 10.4)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}%"))
        ax.set_xlabel(r"window length $d$")
        ax.set_ylabel("common-suffix error")
        ax.text(.045, .95, letter, transform=ax.transAxes, va="top", weight="bold", fontsize=10.5)
        ax.text(.95, .95, rf"$\gamma_T={gamma:g}$", transform=ax.transAxes, ha="right", va="top")
        handles = [Line2D([], [], color=color, marker=marker, linestyle=linestyle,
                          markerfacecolor="white" if series == "history" else color,
                          markersize=3.7, linewidth=1.1, label=label)
                   for series, label, color, marker, linestyle in SERIES]
        if gamma == 1.0:
            legend = ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=1,
                               facecolor="white", edgecolor="black", borderpad=.35,
                               handlelength=1.5, labelspacing=.3)
            _style_legend(legend)
    pdf = output / "fig04_long_context_n5.pdf"
    png = output / "fig04_long_context_n5.png"
    fig.savefig(pdf, metadata=_metadata("Five-qubit long-context follow-up"))
    fig.savefig(png, dpi=320)
    plt.close(fig)
    csv = output / "fig04_long_context_n5_source.csv"
    source.to_csv(csv, index=False)
    artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (pdf, png, csv)}
    (output / "figure_sha256.json").write_text(json.dumps(artifacts, indent=2, sort_keys=True) + "\n")
    return pdf, png, csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for artifact in render(validated_source(args.results, args.analysis), args.out):
        print(artifact)
