"""Strict root-block analysis for the fresh continuous-input robustness arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_pra_extension import bootstrap_median


ROOTS = tuple(f"cont{index:02d}" for index in range(1, 13))
SIZES = (2, 3)
FAMILIES = ("local_loss", "dephasing", "collective_loss")
PANELS = ("canonical_xyz_w2", "coherence_xy_w1", "population_z_w1")
WINDOWS = (1, 4, 8)
KEYS = (
    "root_id",
    "teacher_n_qubits",
    "dissipator_family",
    "reference_local_loss_gamma",
    "observable_panel",
    "window",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_continuous_profile(
    results: pd.DataFrame, run_summary: dict[str, Any]
) -> None:
    required = set(KEYS) | {
        "target_variance",
        "history_mse",
        "history_nrmse",
        "transfer_predicted_history_mse",
        "max_transfer_prediction_error",
        "tracezero_bound_slack",
        "sample_span_bound_slack",
        "reference_jump_strength",
        "matched_jump_strength",
    }
    if missing := required - set(results):
        raise ValueError(f"continuous table missing columns: {sorted(missing)}")
    expected = set(product(ROOTS, SIZES, FAMILIES, (4.0,), PANELS, WINDOWS))
    observed = {
        (
            str(row.root_id),
            int(row.teacher_n_qubits),
            str(row.dissipator_family),
            float(row.reference_local_loss_gamma),
            str(row.observable_panel),
            int(row.window),
        )
        for row in results.loc[:, list(KEYS)].itertuples(index=False)
    }
    if len(results) != len(expected) or observed != expected:
        raise ValueError("continuous table does not match the exact Cartesian profile")
    if results.duplicated(list(KEYS)).any():
        raise ValueError("continuous table contains duplicate cells")
    numeric = results.select_dtypes(include=[np.number])
    if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
        raise ValueError("continuous table contains nonfinite numeric values")
    config = run_summary.get("config", {})
    if config.get("quantize_levels", "missing") is not None:
        raise ValueError("run summary does not certify unsnapped continuous inputs")
    if tuple(config.get("n_qubits", ())) != SIZES:
        raise ValueError("continuous run-summary size axis mismatch")
    if tuple(config.get("windows", ())) != WINDOWS:
        raise ValueError("continuous run-summary window axis mismatch")
    if tuple(config.get("gammas", ())) != (4.0,):
        raise ValueError("continuous run-summary rate axis mismatch")
    max_transfer = float(results["max_transfer_prediction_error"].max())
    max_risk = float(
        np.max(
            np.abs(
                results["history_mse"]
                - results["transfer_predicted_history_mse"]
            )
        )
    )
    max_strength = float(
        np.max(
            np.abs(
                results["reference_jump_strength"]
                - results["matched_jump_strength"]
            )
        )
    )
    if max_transfer > 3e-12 or max_risk > 3e-12 or max_strength > 1e-12:
        raise RuntimeError("continuous transfer or matching invariant failed")
    if (
        float(results["tracezero_bound_slack"].min()) < -3e-12
        or float(results["sample_span_bound_slack"].min()) < -3e-12
    ):
        raise RuntimeError("continuous observable bound failed")


def analyze_continuous(
    results: pd.DataFrame, run_summary: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_continuous_profile(results, run_summary)
    liveness_floor = 1e-8
    work = results.copy()
    work["live"] = work["target_variance"] >= liveness_floor
    paired = work.pivot(
        index=["root_id", "teacher_n_qubits", "dissipator_family", "observable_panel"],
        columns="window",
        values=["history_nrmse", "target_variance"],
    )
    paired = paired[
        (paired[("target_variance", 1)] >= liveness_floor)
        & (paired[("target_variance", 8)] >= liveness_floor)
    ]
    endpoint = pd.DataFrame(
        {
            "history_nrmse_d1": paired[("history_nrmse", 1)],
            "history_nrmse_d8": paired[("history_nrmse", 8)],
        }
    ).reset_index()
    endpoint["gain_d1_to_d8"] = (
        1.0 - endpoint["history_nrmse_d8"] / endpoint["history_nrmse_d1"]
    )
    roots = (
        endpoint.groupby(["root_id", "dissipator_family"], sort=True)
        .agg(
            paired_live_cells=("gain_d1_to_d8", "size"),
            median_history_nrmse_d1=("history_nrmse_d1", "median"),
            median_history_nrmse_d8=("history_nrmse_d8", "median"),
            gain_d1_to_d8=("gain_d1_to_d8", "median"),
        )
        .reset_index()
    )
    effects: dict[str, Any] = {}
    for offset, (family, group) in enumerate(
        roots.groupby("dissipator_family", sort=True)
    ):
        effects[str(family)] = {
            "median_history_nrmse_d1": float(
                np.median(group["median_history_nrmse_d1"])
            ),
            "median_history_nrmse_d8": float(
                np.median(group["median_history_nrmse_d8"])
            ),
            "fractional_reduction": bootstrap_median(
                group["gain_d1_to_d8"], seed=28_082_800 + offset
            ),
            "paired_live_cells_total": int(group["paired_live_cells"].sum()),
        }
    liveness = [
        {
            "dissipator_family": str(family),
            "live_rows": int(group["live"].sum()),
            "total_rows": int(len(group)),
            "live_fraction": float(group["live"].mean()),
        }
        for family, group in work.groupby("dissipator_family", sort=True)
    ]
    return roots, {
        "schema_version": 1,
        "study": "continuous_input_observable_memory_robustness",
        "status": "EXPLORATORY_POST_CONFIRMATORY",
        "claim_boundary": (
            "continuous inputs, N=2/3, matched gamma=4, finite sampled suffix panel; "
            "not a uniform contraction theorem or hardware test"
        ),
        "counts": {"rows": int(len(work)), "roots": int(work["root_id"].nunique())},
        "liveness": {"target_variance_floor": liveness_floor, "by_dissipator": liveness},
        "root_block_effects": effects,
        "numerical_audit": {
            "maximum_transfer_output_error": float(
                work["max_transfer_prediction_error"].max()
            ),
            "maximum_transfer_risk_error": float(
                np.max(
                    np.abs(
                        work["history_mse"]
                        - work["transfer_predicted_history_mse"]
                    )
                )
            ),
            "minimum_tracezero_bound_slack": float(
                work["tracezero_bound_slack"].min()
            ),
            "minimum_sample_span_bound_slack": float(
                work["sample_span_bound_slack"].min()
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observable", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    results = pd.read_csv(args.observable)
    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    expected_hash = run_summary.get("artifact_sha256", {}).get(args.observable.name)
    if expected_hash != sha256(args.observable):
        raise ValueError("observable CSV does not match its run summary hash")
    roots, summary = analyze_continuous(results, run_summary)
    root_path = output / "continuous_root_effects.csv"
    summary_path = output / "continuous_analysis.json"
    roots.to_csv(root_path, index=False, float_format="%.17g")
    summary["source_artifacts"] = {
        str(args.observable): sha256(args.observable),
        str(args.run_summary): sha256(args.run_summary),
    }
    summary["artifacts"] = {root_path.name: sha256(root_path)}
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
