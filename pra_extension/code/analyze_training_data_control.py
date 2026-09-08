"""Validate and summarize the separately frozen training-data follow-up."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from run_training_data_control import (
    config_file_sha256, read_config, validate_coverage, verify_freeze,
)


def read_results(results: Path, config_path: Path, freeze_path: Path) -> tuple[pd.DataFrame, dict]:
    """Use complete root files, not potentially duplicated invocation summaries."""
    config = read_config(config_path)
    freeze = verify_freeze(config_path, freeze_path)
    tables, sources, seen, prefix_stability = [], {}, set(), []
    for path in sorted(Path(results).rglob("metrics.csv")):
        manifest_path = path.with_name("manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"metrics lack a manifest: {path}")
        manifest = json.loads(manifest_path.read_text())
        if "root_id" not in manifest:
            continue
        root_id = manifest["root_id"]
        if root_id in seen:
            raise ValueError(f"duplicate root: {root_id}")
        seen.add(root_id)
        if manifest.get("profile") != "production" or manifest.get("config_sha256") != freeze["config_sha256"]:
            raise ValueError("root is not from this frozen production configuration")
        if manifest.get("metrics_sha256") != config_file_sha256(path):
            raise ValueError("root metric hash mismatch")
        invocation_path = path.parent.parent / "manifest.json"
        invocation = json.loads(invocation_path.read_text())
        if invocation.get("status") != "COMPLETED" or invocation.get("production_freeze") != freeze:
            raise ValueError("root invocation is incomplete or uses another freeze")
        recorded = {r["root_id"]: r["manifest_sha256"] for r in invocation["roots"]}
        if recorded.get(root_id) != config_file_sha256(manifest_path):
            raise ValueError("root manifest hash mismatch")
        frame = pd.read_csv(path, float_precision="round_trip")
        if set(frame["root_id"]) != {root_id} or set(frame["profile"]) != {"production"}:
            raise ValueError("CSV root/profile differs from manifest")
        sources[str(path.resolve())] = config_file_sha256(path)
        for anchor in manifest["anchors"]:
            prefix_stability.append({"root_id": root_id, "anchor_id": anchor["anchor"]["id"],
                                     **anchor["reference_prefix_stability"]})
        tables.append(frame)
    if not tables:
        raise ValueError("no completed production root data")
    frame = pd.concat(tables, ignore_index=True)
    coverage = validate_coverage(frame, config, config["roots"])
    for column in ("reference_mean", "panel_empirical_mean", "panel_exact_mean", "test"):
        expected = np.sqrt(frame[f"{column}_mse"] / frame["target_variance"])
        if not np.allclose(expected, frame[f"{column}_nrmse"], atol=1e-12, rtol=1e-10):
            raise ValueError(f"inconsistent fixed normalization: {column}")
    if frame["identity_error"].max() > 1e-10:
        raise ValueError("finite-panel MSE identity failed")
    return frame, {"freeze": freeze, "coverage": coverage, "source_sha256": sources,
                   "reference_prefix_stability": prefix_stability}


def paired_summary(low: np.ndarray, high: np.ndarray, *, seed: int, replicates: int) -> dict:
    """Positive differences mean less error with more data; ties are failures."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    if low.ndim != 1 or low.shape != high.shape or len(low) == 0:
        raise ValueError("paired vectors must have the same nonempty one-dimensional shape")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low < 0) or np.any(high < 0):
        raise ValueError("paired errors must be finite and nonnegative")
    difference = low - high
    indices = np.random.default_rng(seed).integers(0, len(low), size=(replicates, len(low)))
    interval = np.quantile(np.median(difference[indices], axis=1), [0.025, 0.975])
    positive = int(np.count_nonzero(difference > 0))
    probability = sum(comb(len(low), k) for k in range(positive, len(low) + 1)) / 2**len(low)
    result = {
        "roots": len(low), "positive_roots": positive,
        "median_decrease": float(np.median(difference)),
        "decrease_ci95": interval.tolist(),
        "median_low": float(np.median(low)), "median_high": float(np.median(high)),
        "one_sided_sign_p": probability,
        "median_relative_decrease": None, "relative_decrease_ci95": None,
    }
    if np.all(low > 0):
        relative = difference / low
        result["median_relative_decrease"] = float(np.median(relative))
        result["relative_decrease_ci95"] = np.quantile(
            np.median(relative[indices], axis=1), [0.025, 0.975]).tolist()
    return result


def holm_adjust(probabilities: list[float]) -> list[float]:
    values = np.asarray(probabilities, float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    maximum = 0.0
    for rank, index in enumerate(order):
        maximum = max(maximum, (len(values) - rank) * values[index])
        adjusted[index] = min(maximum, 1.0)
    return adjusted.tolist()


def summarize(frame: pd.DataFrame, config: dict) -> dict:
    primary, secondary, calibration = [], [], []
    first, last = config["training_rows"][0], config["training_rows"][-1]
    roots = [root["root_id"] for root in config["roots"]]
    for anchor in config["anchors"]:
        subset = frame[frame["anchor_id"] == anchor["id"]]
        for (model, budget), values in subset.groupby(["model", "snapshot_budget"], sort=True):
            indexed = values.set_index(["root_id", "training_rows"])
            before = indexed.loc[[(root, first) for root in roots]]
            after = indexed.loc[[(root, last) for root in roots]]
            for metric in ("reference_mean", "test"):
                result = paired_summary(
                    before[f"{metric}_mse"].to_numpy() / before["target_variance"].to_numpy(),
                    after[f"{metric}_mse"].to_numpy() / after["target_variance"].to_numpy(),
                    seed=config["analysis_seed"], replicates=config["bootstrap_replicates"],
                )
                result.update(anchor_id=anchor["id"], model=model, snapshot_budget=int(budget),
                              metric=f"{metric}_nmse", training_low=first, training_high=last)
                (primary if model == "ptd_parallel" and budget == 4 and metric == "reference_mean"
                 else secondary).append(result)
        final = subset[(subset["training_rows"] == last) & (subset["model"] == "ptd_parallel")
                       & (subset["snapshot_budget"] == 4)].sort_values("root_id")
        difference = final["panel_empirical_mean_mse"] - final["panel_exact_mean_mse"]
        calibration.append({
            "anchor_id": anchor["id"], "roots": len(final),
            "median_mean_sampling_nrmse": float(final["panel_mean_sampling_nrmse"].median()),
            "median_empirical_minus_exact_residual_nmse": float(
                np.median(difference / final["target_variance"])),
            "negative_realized_residual_differences": int((difference < 0).sum()),
            "scope": "same 32 IID windows, four prefixes; signed realized difference includes a cross term",
        })
    for result, adjusted in zip(primary, holm_adjust([r["one_sided_sign_p"] for r in primary]), strict=True):
        result["holm_p_two_primary_anchors"] = adjusted
    return {"primary": primary, "secondary": secondary, "finite_prefix_panel_diagnostic": calibration}


def render(frame: pd.DataFrame, config: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.labelsize": 10, "axes.linewidth": 0.7,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.055, 3.38), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.18, top=0.77, wspace=0.34)
    styles = [("ptd_parallel", 1, "#7AD151", "o", "QELM: 1 checkpoint"),
              ("ptd_parallel", 2, "#2A788E", "s", "QELM: 2 checkpoints"),
              ("ptd_parallel", 4, "#482475", "D", "QELM: 4 checkpoints"),
              ("poly2_delay", 0, "#000000", "^", "Quadratic delay")]
    positive_values = frame["reference_mean_nrmse"].to_numpy()
    if np.any(positive_values <= 0):
        raise ValueError("log figure requires positive residuals; use a documented linear-scale alternative")
    lower = 10 ** np.floor(np.log10(positive_values.min()))
    upper = 10 ** np.ceil(np.log10(positive_values.max()))
    for index, (ax, anchor) in enumerate(zip(axes, config["anchors"], strict=True)):
        subset = frame[frame["anchor_id"] == anchor["id"]]
        for model, budget, color, marker, _ in styles:
            selected = subset[(subset["model"] == model) & (subset["snapshot_budget"] == budget)]
            groups = selected.groupby("training_rows")["reference_mean_nrmse"]
            median, low, high = groups.median(), groups.quantile(.1), groups.quantile(.9)
            ax.errorbar(median.index, median, yerr=[median-low, high-median],
                        color=color, marker=marker, markersize=4.4, linewidth=1.05,
                        linestyle="--" if model == "poly2_delay" else "-",
                        markerfacecolor="white" if model == "poly2_delay" else color,
                        markeredgewidth=.8, capsize=2.2, elinewidth=.65, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(config["training_rows"], [f"{x:,}" for x in config["training_rows"]])
        ax.set_xlim(config["training_rows"][0]/1.12, config["training_rows"][-1]*1.12)
        ax.set_yscale("log")
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{100*x:g}%"))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", axis="both", color="#D9D9D9", linestyle=(0, (4, 3)), linewidth=.55)
        ax.grid(True, which="minor", axis="y", color="#EEEEEE", linestyle=":", linewidth=.4)
        ax.set_axisbelow(True)
        ax.set_xlabel("Training rows")
        ax.set_title(f"{anchor['teacher_n_qubits']}-qubit QRC, $d={anchor['window']}$, "
                     f"$\\gamma_T={anchor['gamma']:g}$", fontsize=9.5, pad=7)
        ax.text(.04, .96, "ab"[index], transform=ax.transAxes, va="top", fontweight="bold", fontsize=10)
        ax.set_box_aspect(1)
    axes[0].set_ylabel("Conditional-mean residual NRMSE")
    handles = [Line2D([], [], color=color, marker=marker, markersize=4.5, linewidth=1,
                      linestyle="--" if model == "poly2_delay" else "-",
                      markerfacecolor="white" if model == "poly2_delay" else color, label=label)
               for model, _, color, marker, label in styles]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995), ncol=2,
                        fontsize=8.5, frameon=True, fancybox=True, edgecolor="black",
                        facecolor="white", framealpha=1, handlelength=1.5, columnspacing=1.8)
    legend.get_frame().set_linewidth(.7)
    fig.savefig(output / "training_data_control.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / "training_data_control.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("analysis output must be absent or empty")
    config = read_config(args.config)
    frame, provenance = read_results(args.results, args.config, args.freeze)
    summary = summarize(frame, config)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["anchor_id", "model", "snapshot_budget", "training_rows", "root_id"]).to_csv(
        args.output / "training_data_control_source.csv", index=False, float_format="%.17g")
    render(frame, config, args.output)
    summary.update(schema_version=1, study=config["study"], status="COMPLETE_FOLLOWUP",
                   provenance=provenance, independent_unit="root block; anchors analyzed separately",
                   inference_scope="fixed tested models and finite-prefix input law; no pure approximation-floor claim")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    caption = ("Training-data follow-up. Twelve fresh roots per anchor; exact features and fixed "
               "QRC/QELM architectures. Curves show root medians and 10th–90th percentile whiskers "
               "of residual NRMSE to the exact conditional mean for an IID 128-step prefix, evaluated "
               "on 1,024 independent windows. Errors use the same streaming-test target variance "
               "within each root and anchor. Final readouts use the indicated training rows plus "
               "512 validation rows; standardization uses training rows only. Checkpoint counts "
               "give 18/36/72 features in (a), 30/60/120 in (b). Connecting segments are guides, "
               "not additional observations. This is distinct from worst-case D_d/2 and from "
               "an optimal readout within the feature family.\n")
    (args.output / "CAPTION.md").write_text(caption)
    files = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / "artifact_sha256.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps({"primary": summary["primary"], "figure": str(args.output / "training_data_control.pdf")}, indent=2))


if __name__ == "__main__":
    main()
