"""Strict root-block analysis of the fresh four-qubit-teacher shot arm."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_pra_extension import (
    SHOT_NUMERIC_COLUMNS,
    SHOT_STRUCTURAL_NULL_COLUMNS,
    _parse_integer_list,
    _require_columns,
    _require_finite_numeric,
    bootstrap_median,
    select_budget_on_validation,
    sha256,
)


ROOTS = tuple(f"shotn4_{index:02d}" for index in range(1, 13))
BUDGETS = (1, 2, 4)
TOTALS = (300, 1_200, 4_800, 19_200)
SNAPSHOTS = {1: [3], 2: [1, 3], 4: [0, 1, 2, 3]}
FEATURES = {1: 18, 2: 36, 4: 72}


def validate_n4_shot_profile(results: pd.DataFrame) -> None:
    required = set(SHOT_NUMERIC_COLUMNS) | {
        "root_id",
        "snapshot_indices_zero_based",
        "allocation_mode",
        "allocation_json",
        "intercept_only_selected",
        "monte_carlo_entropy",
    }
    _require_columns(results, required, label="N4 finite-shot")
    _require_finite_numeric(
        results,
        SHOT_NUMERIC_COLUMNS,
        label="N4 finite-shot",
        allow_nan=SHOT_STRUCTURAL_NULL_COLUMNS,
    )
    if not bool((results["teacher_n_qubits"] == 4).all()):
        raise ValueError("N4 finite-shot teacher size differs from four")
    if not bool((results["teacher_gamma"] == 4.0).all()):
        raise ValueError("N4 finite-shot teacher gamma differs from four")
    if not bool((results["student_n_qubits"] == 6).all()):
        raise ValueError("N4 finite-shot student size differs from six")
    if not bool((results["window"] == 6).all()):
        raise ValueError("N4 finite-shot window differs from six")

    observed: set[tuple[str, int, str, int | None]] = set()
    entropies: set[tuple[int, ...]] = set()
    for row in results.itertuples(index=False):
        root = str(row.root_id)
        budget = int(row.snapshot_budget)
        mode = str(row.allocation_mode)
        if root not in ROOTS or budget not in BUDGETS:
            raise ValueError("N4 finite-shot root or budget is off grid")
        if int(row.feature_budget) != FEATURES[budget]:
            raise ValueError("N4 finite-shot feature width is inconsistent")
        if int(row.active_settings) != 3 * budget:
            raise ValueError("N4 finite-shot setting count is inconsistent")
        if _parse_integer_list(
            row.snapshot_indices_zero_based,
            label="snapshot_indices_zero_based",
        ) != SNAPSHOTS[budget]:
            raise ValueError("N4 finite-shot nested snapshots are inconsistent")
        if mode == "noiseless":
            if not pd.isna(row.total_shots) or not pd.isna(row.allocation_json):
                raise ValueError("N4 noiseless row carries shot fields")
            if not pd.isna(row.monte_carlo_entropy) or int(row.mc_repeats) != 0:
                raise ValueError("N4 noiseless row carries Monte Carlo fields")
            total: int | None = None
        elif mode in {"equal", "optimal"}:
            total_value = float(row.total_shots)
            if not total_value.is_integer() or int(total_value) not in TOTALS:
                raise ValueError("N4 finite-shot total is off grid")
            total = int(total_value)
            allocation = _parse_integer_list(row.allocation_json, label="allocation_json")
            if (
                len(allocation) != int(row.active_settings)
                or any(value < 1 for value in allocation)
                or sum(allocation) != total
            ):
                raise ValueError("N4 allocation is not positive and budget preserving")
            if mode == "equal" and max(allocation) - min(allocation) > 1:
                raise ValueError("N4 equal allocation is not balanced")
            entropy = tuple(
                _parse_integer_list(row.monte_carlo_entropy, label="monte_carlo_entropy")
            )
            if entropy in entropies:
                raise ValueError("N4 Monte Carlo entropy tuple is duplicated")
            entropies.add(entropy)
            if int(row.mc_repeats) != 16:
                raise ValueError("N4 finite row has wrong Monte Carlo repeat count")
        else:
            raise ValueError("N4 allocation mode is off grid")
        key = (root, budget, mode, total)
        if key in observed:
            raise ValueError("N4 finite-shot Cartesian key is duplicated")
        observed.add(key)

    expected: set[tuple[str, int, str, int | None]] = set()
    for root, budget in product(ROOTS, BUDGETS):
        expected.add((root, budget, "noiseless", None))
        for mode, total in product(("equal", "optimal"), TOTALS):
            expected.add((root, budget, mode, total))
    if observed != expected or len(results) != 324:
        raise ValueError("N4 finite-shot table differs from the exact 324-cell profile")
    if len(entropies) != 12 * 3 * 4 * 2:
        raise ValueError("N4 Monte Carlo entropy ledger is incomplete")
    if not np.allclose(
        results["test_expected_total_mse"],
        results["test_noiseless_mse"] + results["test_expected_shot_penalty_mse"],
        atol=2e-12,
        rtol=0.0,
    ):
        raise ValueError("N4 finite-shot test MSE identity failed")
    if not np.allclose(
        results["validation_expected_objective_mse"],
        results["validation_noiseless_mse"]
        + results["validation_expected_shot_penalty_mse"],
        atol=2e-12,
        rtol=0.0,
    ):
        raise ValueError("N4 finite-shot validation identity failed")


def analyze_n4_shots(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_n4_shot_profile(results)
    noiseless = results[results["allocation_mode"].eq("noiseless")].copy()
    finite = results[results["allocation_mode"].ne("noiseless")].copy()

    noiseless_wide = noiseless.pivot(
        index="root_id", columns="snapshot_budget", values="test_noiseless_nrmse"
    )
    noiseless_gain = 1.0 - noiseless_wide[4] / noiseless_wide[1]

    fixed_wide = finite[finite["allocation_mode"].eq("optimal")].pivot(
        index=["root_id", "total_shots"],
        columns="snapshot_budget",
        values="test_expected_total_nrmse",
    )
    fixed_wide["gain_B72_vs_B18"] = 1.0 - fixed_wide[4] / fixed_wide[1]
    fixed_wide["gain_B72_vs_B36"] = 1.0 - fixed_wide[4] / fixed_wide[2]

    allocation_wide = finite.pivot(
        index=["root_id", "snapshot_budget", "total_shots"],
        columns="allocation_mode",
        values="test_expected_total_nrmse",
    )
    allocation_wide["gain_optimal_vs_equal"] = (
        1.0 - allocation_wide["optimal"] / allocation_wide["equal"]
    )
    allocation_roots = (
        allocation_wide["gain_optimal_vs_equal"]
        .groupby(["root_id", "total_shots"])
        .median()
    )

    selected = select_budget_on_validation(finite)
    selected_counts = [
        {
            "allocation_mode": str(key[0]),
            "total_shots": int(key[1]),
            "snapshot_budget": int(key[2]),
            "feature_budget": int(key[2]) * 18,
            "roots": int(count),
        }
        for key, count in selected.groupby(
            ["allocation_mode", "total_shots", "snapshot_budget"], sort=True
        ).size().items()
    ]
    surface = [
        {
            "allocation_mode": str(key[0]),
            "total_shots": int(key[1]),
            "snapshot_budget": int(key[2]),
            "feature_budget": int(key[2]) * 18,
            "median_expected_test_nrmse": float(
                np.median(group["test_expected_total_nrmse"])
            ),
        }
        for key, group in finite.groupby(
            ["allocation_mode", "total_shots", "snapshot_budget"], sort=True
        )
    ]
    fixed_effects = {
        f"Q{total}": bootstrap_median(
            fixed_wide.xs(total, level="total_shots")["gain_B72_vs_B18"],
            seed=28_082_900 + offset,
        )
        for offset, total in enumerate(TOTALS)
    }
    fixed_effects_vs_b36 = {
        f"Q{total}": bootstrap_median(
            fixed_wide.xs(total, level="total_shots")["gain_B72_vs_B36"],
            seed=28_082_905 + offset,
        )
        for offset, total in enumerate(TOTALS)
    }
    allocation_effects = {
        f"Q{total}": bootstrap_median(
            allocation_roots.xs(total, level="total_shots"),
            seed=28_082_910 + offset,
        )
        for offset, total in enumerate(TOTALS)
    }
    positive = finite["test_expected_shot_penalty_mse"] > 0
    mc_ratio = (
        finite.loc[positive, "mc_frozen_penalty_mean_mse"]
        / finite.loc[positive, "test_expected_shot_penalty_mse"]
    )
    root_table = fixed_wide.reset_index()
    return root_table, {
        "schema_version": 1,
        "study": "finite_shots_n4_teacher_robustness",
        "status": "EXPLORATORY_POST_CONFIRMATORY",
        "claim_boundary": (
            "four-qubit local-loss teachers and one fixed six-qubit QELM; "
            "exact teacher and training features, noisy test features only"
        ),
        "counts": {"rows": 324, "roots": 12, "teacher_sizes": [4]},
        "noiseless_B72_vs_B18": bootstrap_median(
            noiseless_gain, seed=28_082_899
        ),
        "finite_shot_surface": surface,
        "validation_selected_budget_counts": selected_counts,
        "fixed_B72_vs_B18": fixed_effects,
        "fixed_B72_vs_B36": fixed_effects_vs_b36,
        "optimal_vs_equal": allocation_effects,
        "monte_carlo_to_exact_penalty_ratio": {
            "median": float(np.median(mc_ratio)),
            "minimum": float(np.min(mc_ratio)),
            "maximum": float(np.max(mc_ratio)),
            "repeats_per_cell": 16,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    results = pd.read_csv(args.shots)
    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    expected_hash = run_summary.get("artifact_sha256", {}).get(args.shots.name)
    if expected_hash != sha256(args.shots):
        raise ValueError("N4 shot CSV does not match its run summary hash")
    roots, summary = analyze_n4_shots(results)
    roots_path = output / "n4_shot_root_effects.csv"
    summary_path = output / "n4_shot_analysis.json"
    roots.to_csv(roots_path, index=False, float_format="%.17g")
    summary["source_artifacts"] = {
        str(args.shots): sha256(args.shots),
        str(args.run_summary): sha256(args.run_summary),
    }
    summary["artifacts"] = {roots_path.name: sha256(roots_path)}
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
