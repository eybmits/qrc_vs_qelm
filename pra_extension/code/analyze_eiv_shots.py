"""Strict root-block analysis of the all-splits-noisy N=4 EIV shot arm."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analyze_pra_extension import bootstrap_median, sha256
from finite_shots_eiv import EIV_NAMESPACE_CODE, EIV_SPLIT_CODES


ROOTS = tuple(f"eivn4_{index:02d}" for index in range(1, 13))
BUDGETS = (1, 2, 4)
TOTALS = (300, 1_200, 4_800, 19_200)
REPLICATES = tuple(range(16))
SNAPSHOTS = {1: [3], 2: [1, 3], 4: [0, 1, 2, 3]}
FEATURES = {1: 18, 2: 36, 4: 72}
ALPHA_LABELS = (
    "0",
    "1e-10",
    "1e-08",
    "1e-06",
    "0.0001",
    "0.01",
    "1",
    "100",
    "10000",
    "1000000",
    "+inf",
)
KEYS = ("root_id", "total_shots", "replicate", "snapshot_budget")
NUMERIC_COLUMNS = (
    "teacher_n_qubits",
    "teacher_gamma",
    "student_n_qubits",
    "window",
    "train_rows",
    "validation_rows",
    "test_rows",
    "replicate",
    "snapshot_budget",
    "feature_budget",
    "active_settings",
    "total_shots",
    "minimum_setting_shots",
    "maximum_setting_shots",
    "selected_alpha_index",
    "validation_target_variance",
    "validation_mse",
    "validation_nrmse",
    "test_target_variance",
    "test_noisy_feature_mse",
    "test_noisy_feature_nrmse",
    "test_exact_feature_mse_after_noisy_fit",
    "test_exact_feature_nrmse_after_noisy_fit",
    "noiseless_reference_validation_mse",
    "noiseless_reference_test_mse",
    "noiseless_reference_test_nrmse",
)


def _parse_json_list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a serialized JSON list")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{label} must be a serialized JSON list")
    return parsed


def _integer_list(value: object, *, label: str) -> list[int]:
    parsed = _parse_json_list(value, label=label)
    if any(not isinstance(item, int) or isinstance(item, bool) for item in parsed):
        raise ValueError(f"{label} must contain only integers")
    return [int(item) for item in parsed]


def _finite_float_list(value: object, *, label: str) -> np.ndarray:
    parsed = _parse_json_list(value, label=label)
    try:
        result = np.asarray(parsed, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only numbers") from exc
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite numeric vector")
    return result


def _expected_entropy(
    root: str,
    budget: int,
    total_shots: int,
    replicate: int,
    split: str,
) -> list[int]:
    root_index = int(root.rsplit("_", 1)[1])
    return [
        9_630_000 + 100 * root_index + 1,
        EIV_NAMESPACE_CODE,
        4,
        int(budget),
        int(total_shots),
        int(replicate),
        EIV_SPLIT_CODES[split],
    ]


def _alpha_matches(value: object, expected_label: str) -> bool:
    if expected_label == "+inf":
        return str(value) == "+inf"
    try:
        return float(value) == float(expected_label)
    except (TypeError, ValueError):
        return False


def validate_eiv_profile(results: pd.DataFrame) -> dict[str, Any]:
    """Require the exact 2,304-row profile and every selection/seed invariant."""

    required = set(KEYS) | set(NUMERIC_COLUMNS) | {
        "status",
        "snapshot_indices_zero_based",
        "allocation_mode",
        "allocation_json",
        "train_entropy",
        "validation_entropy",
        "test_entropy",
        "alpha_grid_json",
        "selected_alpha",
        "intercept_only_selected",
        "validation_alpha_mse_json",
        "noiseless_reference_alpha",
        "selected_budget_on_validation",
    }
    if missing := required - set(results.columns):
        raise ValueError(f"EIV table missing columns: {sorted(missing)}")
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(results[column], errors="coerce")
        if bool(converted.isna().any()) or bool(
            np.isinf(converted.to_numpy(dtype=float)).any()
        ):
            raise ValueError(f"EIV numeric column {column!r} is not finite")
    if bool(results.duplicated(list(KEYS)).any()):
        raise ValueError("EIV Cartesian key is duplicated")
    observed = set(results.loc[:, KEYS].itertuples(index=False, name=None))
    expected = set(product(ROOTS, TOTALS, REPLICATES, BUDGETS))
    if observed != expected or len(results) != len(expected):
        raise ValueError("EIV table differs from the exact 2,304-cell profile")

    constants = {
        "status": "EXPLORATORY_POST_CONFIRMATORY",
        "teacher_n_qubits": 4,
        "teacher_gamma": 4.0,
        "student_n_qubits": 6,
        "window": 6,
        "train_rows": 128,
        "validation_rows": 48,
        "test_rows": 96,
        "allocation_mode": "equal_fixed_total",
    }
    for column, expected_value in constants.items():
        if not bool(results[column].eq(expected_value).all()):
            raise ValueError(f"EIV constant field {column!r} differs from profile")
    if bool((results["validation_target_variance"] <= 0).any()) or bool(
        (results["test_target_variance"] <= 0).any()
    ):
        raise ValueError("EIV target variance must be positive")
    nonnegative = (
        "validation_mse",
        "validation_nrmse",
        "test_noisy_feature_mse",
        "test_noisy_feature_nrmse",
        "test_exact_feature_mse_after_noisy_fit",
        "test_exact_feature_nrmse_after_noisy_fit",
        "noiseless_reference_validation_mse",
        "noiseless_reference_test_mse",
        "noiseless_reference_test_nrmse",
    )
    if any(bool((results[column] < 0).any()) for column in nonnegative):
        raise ValueError("EIV table contains a negative risk")
    if not np.allclose(
        results["validation_nrmse"] ** 2,
        results["validation_mse"] / results["validation_target_variance"],
        atol=2e-12,
        rtol=0.0,
    ):
        raise ValueError("EIV validation NRMSE identity failed")
    if not np.allclose(
        results["test_noisy_feature_nrmse"] ** 2,
        results["test_noisy_feature_mse"] / results["test_target_variance"],
        atol=2e-12,
        rtol=0.0,
    ):
        raise ValueError("EIV noisy-test NRMSE identity failed")
    if not np.allclose(
        results["test_exact_feature_nrmse_after_noisy_fit"] ** 2,
        results["test_exact_feature_mse_after_noisy_fit"]
        / results["test_target_variance"],
        atol=2e-12,
        rtol=0.0,
    ):
        raise ValueError("EIV exact-feature diagnostic NRMSE identity failed")

    entropy_seen: set[tuple[int, ...]] = set()
    alpha_grid_serialized = json.dumps(list(ALPHA_LABELS), separators=(",", ":"))
    for row in results.itertuples(index=False):
        root = str(row.root_id)
        budget = int(row.snapshot_budget)
        total_shots = int(row.total_shots)
        replicate = int(row.replicate)
        if int(row.feature_budget) != FEATURES[budget]:
            raise ValueError("EIV feature width differs from snapshot budget")
        if int(row.active_settings) != 3 * budget:
            raise ValueError("EIV active-setting count differs from budget")
        if _integer_list(
            row.snapshot_indices_zero_based,
            label="snapshot_indices_zero_based",
        ) != SNAPSHOTS[budget]:
            raise ValueError("EIV nested snapshot set is not canonical")
        allocation = _integer_list(row.allocation_json, label="allocation_json")
        quotient, remainder = divmod(total_shots, 3 * budget)
        expected_allocation = [
            quotient + (1 if index < remainder else 0)
            for index in range(3 * budget)
        ]
        if allocation != expected_allocation:
            raise ValueError("EIV allocation is not the declared equal integer split")
        if (
            int(row.minimum_setting_shots) != min(allocation)
            or int(row.maximum_setting_shots) != max(allocation)
        ):
            raise ValueError("EIV allocation extrema are inconsistent")
        for split in EIV_SPLIT_CODES:
            entropy = _integer_list(
                getattr(row, f"{split}_entropy"),
                label=f"{split}_entropy",
            )
            if entropy != _expected_entropy(
                root, budget, total_shots, replicate, split
            ):
                raise ValueError(f"EIV {split} entropy differs from profile")
            entropy_tuple = tuple(entropy)
            if entropy_tuple in entropy_seen:
                raise ValueError("EIV split entropy is duplicated")
            entropy_seen.add(entropy_tuple)

        if str(row.alpha_grid_json) != alpha_grid_serialized:
            raise ValueError("EIV alpha grid differs from profile")
        alpha_scores = _finite_float_list(
            row.validation_alpha_mse_json,
            label="validation_alpha_mse_json",
        )
        if len(alpha_scores) != len(ALPHA_LABELS):
            raise ValueError("EIV alpha score trace has the wrong length")
        expected_alpha_index = int(np.argmin(alpha_scores))
        if (
            int(row.selected_alpha_index) != expected_alpha_index
            or not _alpha_matches(
                row.selected_alpha, ALPHA_LABELS[expected_alpha_index]
            )
            or not np.isclose(
                float(row.validation_mse),
                float(alpha_scores[expected_alpha_index]),
                atol=2e-15,
                rtol=2e-14,
            )
        ):
            raise ValueError("EIV alpha is not the noisy-validation minimizer")
        if bool(row.intercept_only_selected) != (
            ALPHA_LABELS[expected_alpha_index] == "+inf"
        ):
            raise ValueError("EIV intercept-only flag is inconsistent")

    if len(entropy_seen) != 12 * 3 * 4 * 16 * 3:
        raise ValueError("EIV split entropy ledger is incomplete")

    expected_selected: set[tuple[str, int, int, int]] = set()
    for key, group in results.groupby(
        ["root_id", "total_shots", "replicate"], sort=True
    ):
        winner = group.sort_values(
            ["validation_mse", "snapshot_budget"], kind="mergesort"
        ).iloc[0]
        expected_selected.add((*key, int(winner["snapshot_budget"])))
    observed_selected = set(
        results.loc[
            results["selected_budget_on_validation"], list(KEYS)
        ].itertuples(
            index=False, name=None
        )
    )
    if observed_selected != expected_selected or len(observed_selected) != 12 * 4 * 16:
        raise ValueError("EIV feature budget was not selected by validation only")

    reference_columns = (
        "noiseless_reference_alpha",
        "noiseless_reference_validation_mse",
        "noiseless_reference_test_mse",
        "noiseless_reference_test_nrmse",
    )
    for column in reference_columns:
        counts = results.groupby(["root_id", "snapshot_budget"])[column].nunique()
        if not bool(counts.eq(1).all()):
            raise ValueError("EIV noiseless reference changes across Q or replicate")
    for column in ("validation_target_variance", "test_target_variance"):
        if not bool(results.groupby("root_id")[column].nunique().eq(1).all()):
            raise ValueError("EIV target variance changes within a root")

    return {
        "cartesian_rows": int(len(results)),
        "validation_selected_rows": int(len(observed_selected)),
        "unique_split_entropy_tuples": int(len(entropy_seen)),
        "alpha_trace_checked_rows": int(len(results)),
        "test_columns_excluded_from_selection": True,
        "exact_reference_constant_within_root_budget": True,
    }


def _root_median(
    frame: pd.DataFrame,
    value: str,
    group: Iterable[str],
) -> pd.Series:
    return frame.groupby(list(group), sort=True)[value].median()


def analyze_eiv(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    invariants = validate_eiv_profile(results)
    selected = results[results["selected_budget_on_validation"]].copy()

    fixed = _root_median(
        results,
        "test_noisy_feature_nrmse",
        ("root_id", "total_shots", "snapshot_budget"),
    ).unstack("snapshot_budget")
    fixed.columns = [f"test_nrmse_B{FEATURES[int(value)]}" for value in fixed.columns]
    fixed["gain_B72_vs_B18"] = (
        1.0 - fixed["test_nrmse_B72"] / fixed["test_nrmse_B18"]
    )
    fixed["gain_B72_vs_B36"] = (
        1.0 - fixed["test_nrmse_B72"] / fixed["test_nrmse_B36"]
    )

    selected_root = _root_median(
        selected,
        "test_noisy_feature_nrmse",
        ("root_id", "total_shots"),
    ).rename("selected_policy_test_nrmse")
    baseline = fixed["test_nrmse_B18"].rename("B18_test_nrmse")
    root_effects = fixed.join(selected_root).join(baseline)
    root_effects["selected_policy_gain_vs_B18"] = (
        1.0
        - root_effects["selected_policy_test_nrmse"]
        / root_effects["B18_test_nrmse"]
    )
    selection_fractions = (
        selected.groupby(["root_id", "total_shots", "snapshot_budget"])
        .size()
        .unstack("snapshot_budget", fill_value=0)
        / 16.0
    )
    selection_fractions = selection_fractions.rename(
        columns={budget: f"selected_fraction_B{FEATURES[budget]}" for budget in BUDGETS}
    )
    root_effects = root_effects.join(selection_fractions).reset_index()

    fixed_surface: list[dict[str, Any]] = []
    for total_shots, budget in product(TOTALS, BUDGETS):
        values = fixed.xs(total_shots, level="total_shots")[
            f"test_nrmse_B{FEATURES[budget]}"
        ]
        summary = bootstrap_median(
            values,
            seed=28_082_700 + TOTALS.index(total_shots) * 10 + budget,
        )
        fixed_surface.append(
            {
                "total_shots": int(total_shots),
                "snapshot_budget": int(budget),
                "feature_budget": FEATURES[budget],
                "median_test_nrmse": summary["median"],
                "ci95_low": summary["ci95_low"],
                "ci95_high": summary["ci95_high"],
            }
        )

    selected_counts = [
        {
            "total_shots": int(key[0]),
            "snapshot_budget": int(key[1]),
            "feature_budget": FEATURES[int(key[1])],
            "root_replicates": int(count),
            "of_root_replicates": 12 * 16,
        }
        for key, count in selected.groupby(
            ["total_shots", "snapshot_budget"], sort=True
        ).size().items()
    ]

    fixed_effects = {
        f"Q{total_shots}": bootstrap_median(
            fixed.xs(total_shots, level="total_shots")["gain_B72_vs_B18"],
            seed=28_082_750 + index,
        )
        for index, total_shots in enumerate(TOTALS)
    }
    selected_policy_effects = {
        f"Q{total_shots}": bootstrap_median(
            root_effects.loc[
                root_effects["total_shots"].eq(total_shots),
                "selected_policy_gain_vs_B18",
            ],
            seed=28_082_760 + index,
        )
        for index, total_shots in enumerate(TOTALS)
    }
    selected_policy_nrmse = {
        f"Q{total_shots}": bootstrap_median(
            root_effects.loc[
                root_effects["total_shots"].eq(total_shots),
                "selected_policy_test_nrmse",
            ],
            seed=28_082_770 + index,
        )
        for index, total_shots in enumerate(TOTALS)
    }
    selected_wide = root_effects.pivot(
        index="root_id",
        columns="total_shots",
        values="selected_policy_test_nrmse",
    )
    selected_shot_gain = 1.0 - selected_wide[19_200] / selected_wide[300]

    reference = (
        results.groupby(["root_id", "snapshot_budget"], sort=True)
        .first()["noiseless_reference_test_nrmse"]
        .unstack("snapshot_budget")
    )
    reference_gain = 1.0 - reference[4] / reference[1]
    all_split_degradation: dict[str, Any] = {}
    noise_location_diagnostic: dict[str, Any] = {}
    b72_exact_after_noisy_fit = _root_median(
        results[results["snapshot_budget"].eq(4)],
        "test_exact_feature_nrmse_after_noisy_fit",
        ("root_id", "total_shots"),
    )
    for total_shots in TOTALS:
        noisy_b72 = fixed.xs(total_shots, level="total_shots")["test_nrmse_B72"]
        fractional_excess = noisy_b72 / reference[4] - 1.0
        all_split_degradation[f"Q{total_shots}"] = bootstrap_median(
            fractional_excess,
            seed=28_082_780 + TOTALS.index(total_shots),
        )
        exact_after = b72_exact_after_noisy_fit.xs(
            total_shots, level="total_shots"
        )
        noise_location_diagnostic[f"Q{total_shots}"] = {
            "train_validation_fractional_excess_exact_test_features": (
                bootstrap_median(
                    exact_after / reference[4] - 1.0,
                    seed=28_082_800 + TOTALS.index(total_shots),
                )
            ),
            "test_measurement_fractional_increment_after_noisy_fit": (
                bootstrap_median(
                    noisy_b72 / exact_after - 1.0,
                    seed=28_082_810 + TOTALS.index(total_shots),
                )
            ),
        }

    selection_frequency_roots: list[dict[str, Any]] = []
    for total_shots, budget in product(TOTALS, BUDGETS):
        column = f"selected_fraction_B{FEATURES[budget]}"
        values = root_effects.loc[
            root_effects["total_shots"].eq(total_shots), column
        ]
        selection_frequency_roots.append(
            {
                "total_shots": int(total_shots),
                "feature_budget": FEATURES[budget],
                "median_root_selection_fraction": float(np.median(values)),
                "minimum_root_selection_fraction": float(np.min(values)),
                "maximum_root_selection_fraction": float(np.max(values)),
            }
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "study": "finite_shots_errors_in_variables_n4",
        "status": "EXPLORATORY_POST_CONFIRMATORY",
        "claim_boundary": (
            "four-qubit local-loss teachers and one fixed six-qubit QELM; "
            "exact teacher labels; independent joint-multinomial shot noise "
            "on train, validation, and test QELM features; equal allocation; "
            "simulation rather than hardware"
        ),
        "selection_rule": (
            "alpha and then B are selected only by noisy validation MSE "
            "within each root/Q/replicate; all reported policy test scores "
            "use that frozen choice"
        ),
        "inference_unit": (
            "16 Monte Carlo replicates are reduced to a median within each "
            "root; intervals bootstrap the 12 roots"
        ),
        "counts": {
            "candidate_rows": 2_304,
            "selected_test_rows": 768,
            "roots": 12,
            "replicates_per_root_resource_cell": 16,
        },
        "strict_invariants": invariants,
        "fixed_test_surface": fixed_surface,
        "validation_selected_budget_counts": selected_counts,
        "validation_selection_frequency_by_root": selection_frequency_roots,
        "fixed_B72_vs_B18": fixed_effects,
        "validation_selected_policy_vs_fixed_B18": selected_policy_effects,
        "validation_selected_policy_test_nrmse": selected_policy_nrmse,
        "validation_selected_policy_Q19200_vs_Q300": bootstrap_median(
            selected_shot_gain, seed=28_082_795
        ),
        "noiseless_reference_B72_vs_B18": bootstrap_median(
            reference_gain, seed=28_082_790
        ),
        "B72_all_split_noise_fractional_excess_over_noiseless_reference": (
            all_split_degradation
        ),
        "B72_noise_location_diagnostic": noise_location_diagnostic,
        "selected_alpha_counts": {
            str(key): int(value)
            for key, value in results["selected_alpha"].value_counts().items()
        },
    }
    return root_effects, selected, summary


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
    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    expected_hash = run_summary.get("artifact_sha256", {}).get(args.shots.name)
    if expected_hash != sha256(args.shots):
        raise ValueError("EIV CSV does not match its run-summary hash")
    if run_summary.get("status") != "EXPLORATORY_POST_CONFIRMATORY":
        raise ValueError("EIV run summary has the wrong governance status")
    if run_summary.get("profile") != "full" or run_summary.get("counts") != {
        "rows": 2_304,
        "roots": 12,
        "replicates": 16,
        "candidate_budgets_per_selection": 3,
        "validation_selected_rows": 768,
    }:
        raise ValueError("EIV run summary differs from the full profile")
    stream_audit = run_summary.get("sampling_streams", {})
    if not (
        stream_audit.get("entropy_tuples") == 6_912
        and stream_audit.get("unique_entropy_tuples") == 6_912
        and stream_audit.get("unique_seedsequence_states") == 6_912
        and stream_audit.get("split_streams_are_disjoint") is True
    ):
        raise ValueError("EIV run-summary sampling ledger is incomplete")
    before = run_summary.get("canonical_freeze_before", {})
    after = run_summary.get("canonical_freeze_after", {})
    if not (
        before.get("passed") is True
        and after.get("passed") is True
        and before.get("material_identity_sha256")
        == after.get("material_identity_sha256")
        == "a65b069e206bf881bef03726325c0117454018e2737f75655f08692e1e565ea2"
    ):
        raise ValueError("EIV run did not preserve the canonical material identity")

    results = pd.read_csv(args.shots)
    root_effects, selected, summary = analyze_eiv(results)
    root_path = output / "eiv_root_effects.csv"
    selected_path = output / "eiv_validation_selected_rows.csv"
    summary_path = output / "eiv_analysis.json"
    root_effects.to_csv(root_path, index=False, float_format="%.17g")
    selected.to_csv(selected_path, index=False, float_format="%.17g")
    summary["source_artifacts"] = {
        str(args.shots): sha256(args.shots),
        str(args.run_summary): sha256(args.run_summary),
    }
    summary["artifacts"] = {
        root_path.name: sha256(root_path),
        selected_path.name: sha256(selected_path),
    }
    code_dir = Path(__file__).resolve().parent
    summary["implementation_sha256"] = {
        name: sha256(code_dir / name)
        for name in (
            "finite_shots.py",
            "finite_shots_eiv.py",
            "run_finite_shots_eiv.py",
            "analyze_eiv_shots.py",
        )
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
