#!/usr/bin/env python3
"""Publication view of the fixed 24-root direct-task replication.

Unsmoothed traces use the prospectively fixed first fresh root and first 96
scored rows. Only whole-test NRMSE scores, never disparate target trajectories,
are aggregated. Every rate, horizon, classical control and candidate is retained
in the ledger; this renderer does not select settings, fit models or infer a
worst-case information floor. All units and intervals are explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

import laser_demonstration
import laser_forecast
import narma_memory
import paper_palette as colors
import task_memory_replication as replication

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "results_task_memory_replication_v1"
DEFAULT_OUTPUT = ROOT / "paper/figures/task_memory"
STEM = "fig04_task_memory"
FIGURE_SIZE = (6.221, 3.65)
N_TRACE = 96
FONT_SIZE = 8.0
CAPTION = (
    "Direct task prediction and history restriction. (a) Measured laser input "
    "and next-step targets/predictions at both fast and slow loss. (b) Synthetic iid "
    "NARMA10 driving input and actual target, compared separately with the fitted QRC "
    "and four-input QELM. Traces are the first 96 scored test rows of the first fresh "
    "root, without smoothing or cross-root averaging. (c) Reset/replay with refitted "
    "readouts, full-QRC references, QELM(4), and quadratic NARMA controls. Medians and "
    "10th-90th root percentiles use all 24 fresh roots; they are variability, not "
    "confidence intervals. Laser roots vary couplings on one reused record; NARMA "
    "roots also vary inputs. Errors are test NRMSE relative to training-target SD, "
    "not worst-case bounds. Longer replay uses more evolution steps. Exact "
    "expectations; no quantum-advantage or identical-cost claim."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Percentiles of rootwise scores at identical declared settings."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in ("study", "condition", "model", "window", "horizon"))
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        seeds = sorted(row["seed"] for row in group)
        expected = replication.LASER_ROOTS if key[0] == "laser" else replication.NARMA_ROOTS
        if seeds != list(expected):
            raise ValueError(f"Expected all 24 unique fresh roots for {key}")
        values = np.asarray([row["nrmse"] for row in group], float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid NRMSE score")
        lower, median, upper = np.quantile(values, [.1, .5, .9])
        output.append(dict(zip(("study", "condition", "model", "window", "horizon"), key))
                      | {"n": len(values), "roots": seeds, "median": float(median),
                         "p10": float(lower), "p90": float(upper)})
    return output


def _gate_report(summaries: dict) -> dict:
    primary = {
        "narma_qelm_minus_qrc": summaries["narma"]["paired_contrasts"]["qelm_minus_qrc"],
        "narma_replay4_minus_qrc": summaries["narma"]["paired_contrasts"]["replay_d4_minus_qrc"],
        "laser_fast_h1_qelm_minus_qrc": summaries["laser"]["paired_contrasts"]["gamma4_h1"]["qelm_minus_qrc"],
        "laser_fast_h1_replay4_minus_qrc": summaries["laser"]["paired_contrasts"]["gamma4_h1"]["restricted_d4_minus_qrc"],
    }
    return {key: {"positive_roots": value["positive_seeds"], "total_roots": value["n_seeds"],
                  "mean": value["mean"], "mean_ci95": value["mean_ci95"],
                  "passed": value["positive_seeds"] >= 20 and value["mean_ci95"][0] > 0}
            for key, value in primary.items()}


def _check_scores(arrays, cell: dict, study: str, targets: dict, training_sd: dict) -> None:
    for row in cell["rows"]:
        horizon = row.get("horizon", 1)
        model, window = row["model"], row["window"]
        if study == "laser":
            prefix = {"replay_refit": "replay", "replay_frozen": "frozen"}.get(model, model)
            key = f"{prefix}_d{window}_h{horizon}" if model.startswith("replay") else f"{model}_h{horizon}"
        else:
            key = f"{model}_d{window}" if model in ("replay", "quadratic") else model
        scored = laser_forecast.score_prediction(arrays[key], targets[horizon], training_sd[horizon])
        for metric in ("rmse", "nrmse"):
            if not np.isclose(scored[metric], row[metric], rtol=1e-11, atol=1e-13):
                raise ValueError(f"Saved prediction/score mismatch: {study}/{cell['seed']}/{key}")
    if study == "narma":
        for row in cell["candidate_rows"]:
            key = f"candidate_{row['model']}_gamma{row['gamma']:g}"
            scored = laser_forecast.score_prediction(arrays[key], targets[1], training_sd[1])
            if not np.isclose(scored["nrmse"], row["nrmse"], rtol=1e-11, atol=1e-13):
                raise ValueError("NARMA candidate score mismatch")
        for model in ("qrc", "qelm"):
            candidates = [row for row in cell["candidate_rows"] if row["model"] == model]
            choice = min(candidates, key=lambda row: row["validation_mse"])
            if choice["gamma"] != cell["selection"][model]:
                raise ValueError("NARMA selection is not validation-only")
            np.testing.assert_array_equal(arrays[model], arrays[f"candidate_{model}_gamma{choice['gamma']:g}"])


def load_source(source: Path = DEFAULT_SOURCE) -> dict:
    """Validate frozen configs/hashes, all saved scores, and causal alignment."""
    source = Path(source)
    spec = replication.validate_spec(source)
    if spec["display"] != {"narma_seed": 921000, "laser_seed": 922000, "scored_rows": [0, 96]}:
        raise ValueError("Unexpected prospective display interval")
    summaries = {study: replication.validate_summary(source, study) for study in ("laser", "narma")}
    cfgs = replication.configs()
    laser_path = ROOT / "data/santa_fe_laser/SF_Acont.dat"
    raw_laser = laser_demonstration.load_laser(laser_path)
    _, laser_indices, laser_targets, preprocessing = laser_forecast.forecast_blocks(raw_laser, cfgs["laser"])
    data_sources = {_relative(source / "replication_specification.json"): digest(source / "replication_specification.json"),
                    _relative(laser_path): digest(laser_path)}
    rows, candidates, traces = [], [], {}
    for study, summary in summaries.items():
        data_sources[_relative(source / study / "summary.json")] = digest(source / study / "summary.json")
        for name, hashed in summary["files_sha256"].items():
            data_sources[_relative(source / study / name)] = hashed
        for cell in sorted(summary["results"], key=lambda c: (c["seed"], c.get("gamma", 0))):
            condition = ("fast" if cell["gamma"] == 4 else "slow") if study == "laser" else "selected"
            for row in cell["rows"]:
                rows.append(dict(row) | {"study": study, "condition": condition, "seed": cell["seed"],
                                         "horizon": row.get("horizon", 1),
                                         "gamma": cell.get("gamma", row.get("gamma"))})
            candidates.extend(dict(row) | {"seed": cell["seed"]} for row in cell.get("candidate_rows", []))
            with np.load(source / study / cell["array_file"], allow_pickle=False) as arrays:
                if study == "laser":
                    targets = {h: laser_targets[h][2] for h in cfgs[study].horizons}
                    scales = {h: float(np.std(laser_targets[h][0])) for h in targets}
                    for horizon in targets:
                        np.testing.assert_array_equal(arrays[f"target_h{horizon}"], targets[horizon])
                    indices = laser_indices[2]
                    np.testing.assert_array_equal(arrays["test_indices"], indices)
                    if cell["seed"] == 922000:
                        offset = sum(cfgs[study].block_lengths[:2])
                        trace = traces.setdefault(study, {
                            "seed": cell["seed"], "test_indices": indices[:N_TRACE].tolist(),
                            "target_indices": (indices[:N_TRACE] + 1).tolist(),
                            "input": raw_laser[offset + indices[:N_TRACE]].tolist(),
                            "target": targets[1][:N_TRACE].tolist(),
                            "input_units": "unquantized measured laser units; hardware input is train-scaled and quantized",
                            "output_units": "unquantized measured laser units", "conditions": {},
                        })
                        trace["conditions"][condition] = {"gamma": cell["gamma"],
                            "qrc": arrays["qrc_h1"][:N_TRACE].tolist(), "qelm": arrays["qelm_h1"][:N_TRACE].tolist()}
                else:
                    raw, _, indices_by_split, target_by_split = narma_memory.make_data(cell["seed"], cfgs[study])
                    indices = indices_by_split[2]
                    targets, scales = {1: target_by_split[2]}, {1: float(np.std(target_by_split[0]))}
                    np.testing.assert_array_equal(arrays["test_indices"], indices)
                    np.testing.assert_array_equal(arrays["test_target"], targets[1])
                    np.testing.assert_array_equal(arrays["test_raw_input"], raw[2])
                    if cell["seed"] == 921000:
                        traces[study] = {"seed": cell["seed"], "test_indices": indices[:N_TRACE].tolist(),
                            "target_indices": (indices[:N_TRACE] + 1).tolist(),
                            "input": (cfgs[study].input_scale * raw[2][indices[:N_TRACE]]).tolist(),
                            "target": targets[1][:N_TRACE].tolist(),
                            "qrc": arrays["qrc"][:N_TRACE].tolist(), "qelm": arrays["qelm"][:N_TRACE].tolist(),
                            "selected_gamma": cell["selection"],
                            "input_units": "NARMA recursion drive 0.2*u[t]; hardware receives quantized u[t]",
                            "output_units": "unquantized synthetic NARMA10 target"}
                _check_scores(arrays, cell, study, targets, scales)
    gates = _gate_report(summaries)
    report_path = source / "replication_report.json"
    if report_path.exists():
        if _json(report_path)["direction_checks"] != gates:
            raise ValueError("Replication report disagrees with verified study summaries")
        data_sources[_relative(report_path)] = digest(report_path)
    selected_sources = [Path(__file__), Path(colors.__file__)]
    verification = None
    verification_path = source / "verification.json"
    if verification_path.exists():
        verification = _json(verification_path)
        verifier = Path(__file__).with_name("verify_task_memory_replication.py")
        if (verification["verification_passed"] is not True
                or verification["direction_checks"] != gates
                or verification["specification_sha256"] != digest(source / "replication_specification.json")
                or verification["verifier_sha256"] != digest(verifier)
                or verification["summary_sha256"] != {study: digest(source / study / "summary.json") for study in summaries}):
            raise ValueError("Independent verification does not identify these source results")
        data_sources[_relative(verification_path)] = digest(verification_path)
        selected_sources.append(verifier)
    sources = replication.sources() | {_relative(p): digest(p) for p in selected_sources}
    payload = {"schema_version": 1, "sources": sources, "data_sources": data_sources,
               "replication_specification": spec, "study_summaries": summaries,
               "all_rows": rows, "candidate_rows": candidates, "traces": traces,
               "aggregates": aggregate_rows(rows), "direction_checks": gates,
               "independent_verification": verification,
               "preprocessing": preprocessing, "caption": CAPTION,
               "scope": {
                   "metric": "test RMSE / training-target SD (NRMSE); not worst-case error",
                   "interval": "10th-90th percentiles of 24 rootwise scores, not confidence intervals",
                   "laser": "fresh coupling roots conditional on one reused measured record; both rates retained",
                   "narma": "fresh independent input and coupling roots; no past targets supplied",
                   "traces": "first fresh root, first 96 scored rows; no smoothing or cross-root averaging",
                   "history": "reset same QRC and refit readout; increasing replay uses more evolution steps",
                   "controls": "all h=4, frozen-readout, persistence, training-mean and candidate outcomes retained",
                   "claims": "exact expectations; no quantum advantage, universal superiority or information floor"}}
    validate_payload(payload)
    return payload


def validate_payload(payload: dict) -> None:
    """Portable checks also apply to the checked-in ledger when raw data are absent."""
    if payload["schema_version"] != 1:
        raise ValueError("Unknown figure source schema")
    expected = {"laser": {(c, m, w, h) for c in ("fast", "slow") for h in (1, 4)
                           for m, w in [("qrc", None), ("qelm", 4), ("persistence", 1)]
                           + [(m, w) for m in ("replay_refit", "replay_frozen") for w in (1, 2, 4, 8, 16, 32)]},
                "narma": {("selected", m, w, 1) for m, w in [("qrc", None), ("qelm", 4), ("training_mean", None)]
                           + [("replay", w) for w in (1, 4, 8, 12, 24, 48)]
                           + [("quadratic", w) for w in (4, 12)]}}
    for study in expected:
        observed = {(r["condition"], r["model"], r["window"], r["horizon"])
                    for r in payload["all_rows"] if r["study"] == study}
        if observed != expected[study]:
            raise ValueError(f"Incomplete {study} condition coverage")
    if payload["aggregates"] != aggregate_rows(payload["all_rows"]):
        raise ValueError("Aggregate rows do not match the complete root cohort")
    if len(payload["candidate_rows"]) != 24 * 8:
        raise ValueError("Incomplete NARMA candidates")
    candidate_keys = {(row["seed"], row["model"], row["gamma"]) for row in payload["candidate_rows"]}
    if candidate_keys != {(seed, model, rate) for seed in replication.NARMA_ROOTS
                          for model in ("qrc", "qelm") for rate in (.1, .3, 1., 4.)}:
        raise ValueError("NARMA candidate coverage drift")
    for study, seed in (("laser", 922000), ("narma", 921000)):
        trace = payload["traces"][study]
        if trace["seed"] != seed or trace["test_indices"] != list(range(128, 224)):
            raise ValueError("Unexpected display seed or scored interval")
        if trace["target_indices"] != list(range(129, 225)):
            raise ValueError("Noncausal next-step target alignment")
        arrays = [trace["input"], trace["target"]]
        if study == "laser":
            if set(trace["conditions"]) != {"fast", "slow"}:
                raise ValueError("Both laser rates must remain visible")
            for condition, rate in (("fast", 4), ("slow", .2)):
                if trace["conditions"][condition]["gamma"] != rate:
                    raise ValueError("Unexpected laser loss condition")
                arrays.extend(trace["conditions"][condition][m] for m in ("qrc", "qelm"))
        else:
            arrays.extend(trace[m] for m in ("qrc", "qelm"))
        if any(np.asarray(a).shape != (N_TRACE,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Expected 96 finite unsmoothed trace samples")
    if payload["direction_checks"] != _gate_report(payload["study_summaries"]):
        raise ValueError("Replication gate metadata drift")


def _style() -> None:
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
        "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "mathtext.fontset": "cm", "text.color": colors.INK, "axes.labelcolor": colors.INK,
        "axes.edgecolor": colors.INK, "xtick.color": colors.INK, "ytick.color": colors.INK,
        "axes.linewidth": .65, "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white"})


def _axis(fig, left, bottom, width, height, *, name):
    ax = fig.add_axes([left / FIGURE_SIZE[0], bottom / FIGURE_SIZE[1],
                       width / FIGURE_SIZE[0], height / FIGURE_SIZE[1]], label=name)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="both", color=colors.GRID, linewidth=.4, linestyle=(0, (4, 4)))
    ax.tick_params(pad=2)
    return ax


def _traces(fig, payload):
    axes = {}
    x = np.arange(N_TRACE)
    for study, left in (("laser", .44), ("narma", 2.48)):
        trace = payload["traces"][study]
        width = 1.56 if study == "laser" else 1.61
        ax = _axis(fig, left, 2.59, width, .39, name=f"{study}_input")
        axes[f"{study}_input"] = ax
        ax.plot(x, trace["input"], color=colors.MUTED, linewidth=.65)
        ax.set_ylabel("input", labelpad=3)
        ax.set_xlim(0, 95)
        ax.set_xticks([0, 48, 95], labels=[])
        ax.yaxis.set_major_locator(MaxNLocator(2))
        if study == "narma":
            ax.set_ylim(0, .2)
            ax.set_yticks([0, .2], labels=["0", ".2"])
        sets = [("fast", "fast loss: $\\gamma=4$"), ("slow", "slow loss: $\\gamma=0.2$")] if study == "laser" else [
            ("qrc", "QRC prediction"), ("qelm", "QELM prediction")]
        values = [np.asarray(trace["target"])]
        if study == "laser":
            values.extend(np.asarray(trace["conditions"][c][m]) for c in ("fast", "slow") for m in ("qrc", "qelm"))
        else:
            values.extend(np.asarray(trace[m]) for m in ("qrc", "qelm"))
        lo, hi = min(float(v.min()) for v in values), max(float(v.max()) for v in values)
        padding = max((hi - lo) * .09, .001)
        for (condition, title), bottom in zip(sets, (1.48, .42), strict=True):
            ax = _axis(fig, left, bottom, width, .80, name=f"{study}_{condition}")
            axes[f"{study}_{condition}"] = ax
            ax.plot(x, trace["target"], color=colors.COBALT, linewidth=.95, label="target", zorder=2)
            models = ("qrc", "qelm") if study == "laser" else (condition,)
            for model in models:
                predicted = trace["conditions"][condition][model] if study == "laser" else trace[model]
                ax.plot(x, predicted, color=colors.INK if model == "qrc" else colors.CORAL,
                        linewidth=.8, linestyle=(0, (3, 2)) if model == "qrc" else "-",
                        marker=None if model == "qrc" else "o", markerfacecolor="white",
                        markersize=2.2, markeredgewidth=.5, markevery=8, label=model.upper(), zorder=3)
            ax.set_ylim(lo - padding, hi + padding)
            ax.set_xlim(0, 95)
            ax.set_xticks([0, 48, 95])
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.set_ylabel("output", labelpad=3)
            ax.set_title(title, pad=3)
            if bottom > 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("test step (excerpt)", labelpad=3)
    handles = [Line2D([], [], color=colors.COBALT, lw=.95, label="target"),
               Line2D([], [], color=colors.INK, lw=.8, linestyle=(0, (3, 2)), label="QRC"),
               Line2D([], [], color=colors.CORAL, lw=.8, marker="o", markerfacecolor="white", markersize=3, label="QELM")]
    legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=(2.23 / FIGURE_SIZE[0], 3.20 / FIGURE_SIZE[1]),
                        ncol=3, frameon=False, handlelength=1.8, columnspacing=1.2, handletextpad=.5)
    return axes, legend


def _summary(payload, study, condition, model, horizon=1):
    return sorted((r for r in payload["aggregates"] if (r["study"], r["condition"], r["model"], r["horizon"])
                   == (study, condition, model, horizon)), key=lambda r: r["window"] or 0)


def _history(fig, payload):
    axes = {}
    for study, bottom, height in (("laser", 2.04, 1.00), ("narma", .42, 1.02)):
        ax = _axis(fig, 4.62, bottom, 1.53, height, name=f"{study}_history")
        axes[f"{study}_history"] = ax
        conditions = ("fast", "slow") if study == "laser" else ("selected",)
        windows = [1, 2, 4, 8, 16, 32] if study == "laser" else [1, 4, 8, 12, 24, 48]
        high = 0
        for condition in conditions:
            slow = condition == "slow"
            linestyle = (0, (3, 2)) if slow else "-"
            marker = "s" if slow else "o"
            rows = _summary(payload, study, condition, "replay_refit" if study == "laser" else "replay")
            median, lower, upper = (np.asarray([r[k] for r in rows]) for k in ("median", "p10", "p90"))
            ax.fill_between(windows, lower, upper, color=colors.VIOLET, alpha=.10, linewidth=0)
            ax.plot(windows, median, color=colors.VIOLET, marker=marker, markersize=2.8,
                    markerfacecolor="none" if slow else colors.VIOLET, markeredgewidth=.6,
                    linewidth=.95, linestyle=linestyle, label=f"replay_{condition}")
            ref = _summary(payload, study, condition, "qrc")[0]
            ax.axhspan(ref["p10"], ref["p90"], color=colors.INK, alpha=.045, linewidth=0)
            ax.axhline(ref["median"], color=colors.INK, linestyle=linestyle, linewidth=.85, label=f"qrc_{condition}")
            qelm = _summary(payload, study, condition, "qelm")[0]
            ax.errorbar([4], [qelm["median"]], yerr=[[qelm["median"] - qelm["p10"]], [qelm["p90"] - qelm["median"]]],
                        color=colors.CORAL, marker=marker, markerfacecolor="none" if slow else colors.CORAL,
                        markersize=4, markeredgewidth=.75, capsize=1.8, elinewidth=.7, linestyle="none", label=f"qelm_{condition}", zorder=5)
            high = max(high, float(upper.max()), qelm["p90"], ref["p90"])
        if study == "narma":
            rows = _summary(payload, study, "selected", "quadratic")
            median, lower, upper = (np.asarray([r[k] for r in rows]) for k in ("median", "p10", "p90"))
            ax.errorbar([r["window"] for r in rows], median, yerr=[median - lower, upper - median],
                        color=colors.MUTED, marker="D", markerfacecolor="white", markersize=3.4,
                        markeredgewidth=.8, linestyle=(0, (1, 2)), linewidth=.8, capsize=1.8, elinewidth=.7,
                        label="quadratic", zorder=4)
            high = max(high, float(upper.max()))
        ax.set_xscale("log", base=2)
        ax.set_xlim(.85, windows[-1] * 1.18)
        ax.set_xticks([1, 4, 12, 48] if study == "narma" else [1, 4, 16, 32],
                      labels=["1", "4", "12", "48"] if study == "narma" else ["1", "4", "16", "32"])
        ax.set_ylim(0, high * 1.09)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:g}%"))
        ax.set_ylabel("test NRMSE", labelpad=3)
        ax.set_xlabel("replayed inputs $d$" if study == "narma" else "", labelpad=3)
        ax.text(.98, .97, "Laser" if study == "laser" else "NARMA10", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5)
    handles = [Line2D([], [], color=colors.VIOLET, marker="o", markersize=2.8, lw=.95, label="replay/refit"),
               Line2D([], [], color=colors.INK, lw=.85, label="QRC (full)"),
               Line2D([], [], color=colors.CORAL, marker="o", markersize=3.5, lw=0, label="QELM(4)"),
               Line2D([], [], color=colors.MUTED, marker="D", markerfacecolor="white", markersize=3.2, lw=.7, linestyle=":", label="quadratic")]
    legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=(5.36 / FIGURE_SIZE[0], 1.66 / FIGURE_SIZE[1]),
                        ncol=2, frameon=False, handlelength=1.2, columnspacing=.8, handletextpad=.4, labelspacing=.35)
    rates = [Line2D([], [], color=colors.MUTED, linestyle="-", marker="o", markersize=2.5, lw=.8, label=r"$\gamma=4$"),
             Line2D([], [], color=colors.MUTED, linestyle="--", marker="s", markerfacecolor="white", markersize=2.5, lw=.8, label=r"$\gamma=0.2$")]
    rate_legend = fig.legend(handles=rates, loc="center", bbox_to_anchor=(5.36 / FIGURE_SIZE[0], 3.22 / FIGURE_SIZE[1]),
                             ncol=2, frameon=False, handlelength=1.5, columnspacing=.8, handletextpad=.4)
    return axes, (legend, rate_legend)


def make_figure(payload: dict):
    validate_payload(payload)
    _style()
    fig = plt.figure(figsize=FIGURE_SIZE)
    axes, trace_legend = _traces(fig, payload)
    history_axes, history_legends = _history(fig, payload)
    axes.update(history_axes)
    for letter, title, left in (("a", "Laser forecasting", .44), ("b", "NARMA10", 2.48), ("c", "History restriction", 4.62)):
        fig.text(left / FIGURE_SIZE[0], 3.50 / FIGURE_SIZE[1], letter, fontsize=10, fontweight="bold", va="center")
        fig.text((left + .18) / FIGURE_SIZE[0], 3.50 / FIGURE_SIZE[1], title, fontsize=8.5, va="center")
    return fig, axes, (trace_legend, *history_legends)


def _csv(path: Path, rows: list[dict]) -> None:
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render_payload(payload: dict, output: Path = DEFAULT_OUTPUT) -> dict:
    """Render a validated portable ledger without accessing ignored simulation files."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        fig, _, _ = make_figure(payload)
        fig.savefig(output / f"{STEM}.pdf", metadata={"Title": "Direct task prediction and controlled history restriction",
                    "Creator": "qrc_vs_qelm task-memory renderer", "CreationDate": None, "ModDate": None})
        fig.savefig(output / f"{STEM}.png", dpi=300, metadata={"Software": "qrc_vs_qelm"})
        plt.close(fig)
    _csv(output / f"{STEM}_rows.csv", payload["all_rows"])
    trace_rows = []
    for study, trace in payload["traces"].items():
        for k in range(N_TRACE):
            row = {"study": study, "seed": trace["seed"], "excerpt_step": k,
                   "test_index": trace["test_indices"][k], "target_index": trace["target_indices"][k],
                   "input": trace["input"][k], "target": trace["target"][k]}
            if study == "laser":
                row.update({f"{model}_{condition}": trace["conditions"][condition][model][k]
                            for condition in ("fast", "slow") for model in ("qrc", "qelm")})
            else:
                row.update({model: trace[model][k] for model in ("qrc", "qelm")})
            trace_rows.append(row)
    _csv(output / f"{STEM}_traces.csv", trace_rows)
    ledger = dict(payload) | {"figure_size_inches": list(FIGURE_SIZE), "font_size_pt": FONT_SIZE,
        "palette": {"target": colors.COBALT, "qrc": colors.INK, "qelm": colors.CORAL,
                    "replay": colors.VIOLET, "quadratic": colors.MUTED, "input": colors.MUTED},
        "panels": {"a": "laser input and true next-step forecasting, both fixed loss rates",
                   "b": "synthetic NARMA10 input and true next-step forecasting, one fresh root",
                   "c": "controlled reset/refit history restriction, 24-root NRMSE medians and 10th-90th percentiles"},
        "artifacts": {f"{STEM}{suffix}": digest(output / f"{STEM}{suffix}")
                      for suffix in (".pdf", ".png", "_rows.csv", "_traces.csv")}}
    ledger["sources"] = dict(payload["sources"]) | {_relative(Path(__file__)): digest(Path(__file__))}
    (output / f"{STEM}_source.json").write_text(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return ledger


def render(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> dict:
    return render_payload(load_source(source), output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ledger = render(args.source, args.output)
    print(json.dumps({"output": str(args.output), "direction_checks": ledger["direction_checks"],
                      "figure_size_inches": ledger["figure_size_inches"]}, indent=2))


if __name__ == "__main__":
    main()
