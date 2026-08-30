"""Root-block analysis for the 12-root exploratory PRA stress tests.

The script is intentionally separate from the frozen canonical study.  It
validates the extension outputs, computes paired root-level effects, and emits
content-addressed summary artifacts.  Root blocks, never individual rows or
teacher observables, are the inferential unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OBSERVABLE_ROOTS = tuple(f"pra{index:02d}" for index in range(5, 17))
OBSERVABLE_N = (2, 3, 4)
OBSERVABLE_FAMILIES = ("local_loss", "dephasing", "collective_loss")
OBSERVABLE_GAMMAS = (0.25, 1.0, 4.0)
OBSERVABLE_PANELS = (
    "canonical_xyz_w2",
    "coherence_xy_w1",
    "population_z_w1",
)
OBSERVABLE_WINDOWS = (1, 2, 4, 8)
OBSERVABLE_KEY_COLUMNS = (
    "root_id",
    "teacher_n_qubits",
    "dissipator_family",
    "reference_local_loss_gamma",
    "observable_panel",
    "window",
)
OBSERVABLE_NUMERIC_COLUMNS = (
    "coupling_seed",
    "prefix_seed",
    "suffix_seed",
    "teacher_n_qubits",
    "reference_local_loss_gamma",
    "observable_max_weight",
    "reference_jump_strength",
    "raw_jump_strength",
    "matched_jump_strength",
    "jump_rate_rescale",
    "window",
    "n_suffix_groups",
    "prefix_branches",
    "n_outputs",
    "target_variance",
    "history_mse",
    "history_nrmse",
    "initial_observable_mse",
    "observable_memory_gain",
    "initial_state_hs_mse",
    "final_state_hs_mse",
    "state_hs_memory_gain",
    "transfer_predicted_history_mse",
    "max_transfer_prediction_error",
    "tracezero_observable_gain_mean",
    "tracezero_observable_gain_max",
    "sample_span_observable_gain_mean",
    "sample_span_observable_gain_max",
    "tracezero_upper_mse",
    "sample_span_upper_mse",
    "tracezero_bound_slack",
    "sample_span_bound_slack",
    "max_half_output_diameter",
)

SHOT_ROOTS = tuple(f"shot{index:02d}" for index in range(4, 16))
SHOT_TEACHER_N = (2, 3)
SHOT_BUDGETS = (1, 2, 4)
SHOT_TOTALS = (300, 1_200, 4_800, 19_200)
SHOT_ALLOCATIONS = ("noiseless", "equal", "optimal")
SHOT_KEY_COLUMNS = (
    "root_id",
    "teacher_n_qubits",
    "snapshot_budget",
    "allocation_mode",
    "total_shots",
)
SHOT_STRUCTURAL_NULL_COLUMNS = (
    "total_shots",
    "allocation_json",
    "minimum_setting_shots",
    "maximum_setting_shots",
    "mc_frozen_penalty_mean_mse",
    "mc_frozen_penalty_std_mse",
)
SHOT_NUMERIC_COLUMNS = (
    "teacher_n_qubits",
    "teacher_gamma",
    "student_n_qubits",
    "window",
    "snapshot_budget",
    "feature_budget",
    "active_settings",
    "total_shots",
    "minimum_setting_shots",
    "maximum_setting_shots",
    "selected_alpha",
    "validation_noiseless_mse",
    "validation_expected_shot_penalty_mse",
    "validation_expected_objective_mse",
    "test_target_variance",
    "test_noiseless_mse",
    "test_noiseless_nrmse",
    "test_expected_shot_penalty_mse",
    "test_expected_total_mse",
    "test_expected_total_nrmse",
    "mc_frozen_penalty_mean_mse",
    "mc_frozen_penalty_std_mse",
    "mc_repeats",
)

VISIBILITY_TOLERANCE = 2e-12


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} table missing columns: {sorted(missing)}")


def _require_finite_numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
    allow_nan: Iterable[str] = (),
) -> None:
    allowed = set(allow_nan)
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        nonnumeric = frame[column].notna() & converted.isna()
        if bool(nonnumeric.any()):
            raise ValueError(f"{label} column {column!r} contains nonnumeric values")
        values = converted.to_numpy(dtype=float)
        if bool(np.isinf(values).any()):
            raise ValueError(f"{label} column {column!r} contains infinite values")
        if column not in allowed and bool(np.isnan(values).any()):
            raise ValueError(f"{label} column {column!r} contains missing values")


def _assert_exact_keys(
    frame: pd.DataFrame,
    key_columns: Iterable[str],
    expected: set[tuple[object, ...]],
    *,
    label: str,
) -> None:
    keys = list(key_columns)
    if bool(frame.duplicated(keys).any()):
        duplicate = tuple(frame.loc[frame.duplicated(keys, keep=False), keys].iloc[0])
        raise ValueError(f"{label} table has duplicate Cartesian key {duplicate!r}")
    observed = set(frame.loc[:, keys].itertuples(index=False, name=None))
    if observed != expected:
        missing = sorted(expected - observed, key=repr)[:3]
        extra = sorted(observed - expected, key=repr)[:3]
        raise ValueError(
            f"{label} Cartesian keys differ from the exact v2 profile; "
            f"missing={missing!r}, extra={extra!r}"
        )


def validate_observable_v2_profile(results: pd.DataFrame) -> None:
    """Require the exact 12-root observable-memory v2 Cartesian profile."""

    required = set(OBSERVABLE_KEY_COLUMNS) | set(OBSERVABLE_NUMERIC_COLUMNS) | {
        "observable_axes",
    }
    _require_columns(results, required, label="observable")
    _require_finite_numeric(
        results, OBSERVABLE_NUMERIC_COLUMNS, label="observable"
    )
    expected = set(
        product(
            OBSERVABLE_ROOTS,
            OBSERVABLE_N,
            OBSERVABLE_FAMILIES,
            OBSERVABLE_GAMMAS,
            OBSERVABLE_PANELS,
            OBSERVABLE_WINDOWS,
        )
    )
    _assert_exact_keys(
        results,
        OBSERVABLE_KEY_COLUMNS,
        expected,
        label="observable",
    )
    if len(results) != len(expected):
        raise ValueError("observable table row count differs from the v2 profile")
    if not bool((results["n_suffix_groups"] == 12).all()):
        raise ValueError("observable table has an unexpected suffix-group count")
    if not bool((results["prefix_branches"] == 4).all()):
        raise ValueError("observable table has an unexpected prefix-branch count")

    panel_metadata = {
        "canonical_xyz_w2": ("xyz", 2, lambda n: 3 * n + 3 * n * (n - 1) // 2),
        "coherence_xy_w1": ("xy", 1, lambda n: 2 * n),
        "population_z_w1": ("z", 1, lambda n: n),
    }
    for row in results.itertuples(index=False):
        axes, max_weight, width = panel_metadata[str(row.observable_panel)]
        if (
            str(row.observable_axes) != axes
            or int(row.observable_max_weight) != max_weight
            or int(row.n_outputs) != width(int(row.teacher_n_qubits))
        ):
            raise ValueError("observable panel metadata differs from the v2 profile")
        root_index = int(str(row.root_id)[3:])
        n_qubits = int(row.teacher_n_qubits)
        if (
            int(row.coupling_seed) != 9_100_000 + 100 * root_index + n_qubits
            or int(row.prefix_seed) != 9_110_000 + 100 * root_index + 1
            or int(row.suffix_seed) != 9_120_000 + 100 * root_index + 1
        ):
            raise ValueError("observable seed metadata differs from the v2 profile")
    nonnegative = (
        "target_variance",
        "history_mse",
        "history_nrmse",
        "initial_observable_mse",
        "initial_state_hs_mse",
        "final_state_hs_mse",
        "tracezero_upper_mse",
        "sample_span_upper_mse",
    )
    if any(bool((results[column] < 0).any()) for column in nonnegative):
        raise ValueError("observable table contains a negative risk or variance")
    if bool((results["final_state_hs_mse"] <= 0).any()):
        raise ValueError("measurement visibility requires positive final-state HS MSE")


def _parse_integer_list(value: object, *, label: str) -> list[int]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a serialized integer list")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in parsed
    ):
        raise ValueError(f"{label} must contain only integers")
    return parsed


def validate_shot_v2_profile(results: pd.DataFrame) -> None:
    """Require the exact 12-root finite-shot v2 Cartesian profile."""

    required = set(SHOT_KEY_COLUMNS) | set(SHOT_NUMERIC_COLUMNS) | {
        "snapshot_indices_zero_based",
        "allocation_json",
        "intercept_only_selected",
    }
    _require_columns(results, required, label="finite-shot")
    _require_finite_numeric(
        results,
        SHOT_NUMERIC_COLUMNS,
        label="finite-shot",
        allow_nan=SHOT_STRUCTURAL_NULL_COLUMNS,
    )
    if not bool((results["teacher_gamma"] == 4.0).all()):
        raise ValueError("finite-shot teacher gamma differs from the v2 profile")
    if not bool((results["student_n_qubits"] == 6).all()):
        raise ValueError("finite-shot student size differs from the v2 profile")
    if not bool((results["window"] == 6).all()):
        raise ValueError("finite-shot window differs from the v2 profile")

    normalized_keys: list[tuple[object, ...]] = []
    expected_snapshot_indices = {1: [3], 2: [1, 3], 4: [0, 1, 2, 3]}
    expected_feature_width = {1: 18, 2: 36, 4: 72}
    for row in results.itertuples(index=False):
        mode = str(row.allocation_mode)
        budget = int(row.snapshot_budget)
        if budget not in expected_snapshot_indices:
            raise ValueError("finite-shot snapshot budget differs from the v2 profile")
        if int(row.feature_budget) != expected_feature_width[budget]:
            raise ValueError("finite-shot feature budget differs from snapshot budget")
        if int(row.active_settings) != 3 * budget:
            raise ValueError("finite-shot active-setting count is inconsistent")
        if _parse_integer_list(
            row.snapshot_indices_zero_based,
            label="snapshot_indices_zero_based",
        ) != expected_snapshot_indices[budget]:
            raise ValueError("finite-shot nested snapshot subset is not canonical")

        if mode == "noiseless":
            if not pd.isna(row.total_shots):
                raise ValueError("noiseless rows must not carry a shot budget")
            for column in SHOT_STRUCTURAL_NULL_COLUMNS[1:]:
                if not pd.isna(getattr(row, column)):
                    raise ValueError(
                        f"noiseless row must have null structural field {column!r}"
                    )
            if int(row.mc_repeats) != 0:
                raise ValueError("noiseless rows must have zero MC repeats")
            normalized_total: int | None = None
        elif mode in {"equal", "optimal"}:
            for column in SHOT_STRUCTURAL_NULL_COLUMNS[1:]:
                if pd.isna(getattr(row, column)):
                    raise ValueError(
                        f"finite-shot row has null structural field {column!r}"
                    )
            total_value = float(row.total_shots)
            if not total_value.is_integer() or int(total_value) not in SHOT_TOTALS:
                raise ValueError("finite-shot total budget differs from the v2 profile")
            normalized_total = int(total_value)
            allocation = _parse_integer_list(
                row.allocation_json, label="allocation_json"
            )
            if (
                len(allocation) != int(row.active_settings)
                or any(value < 1 for value in allocation)
                or sum(allocation) != normalized_total
            ):
                raise ValueError("finite-shot allocation is not positive and budget preserving")
            if (
                int(row.minimum_setting_shots) != min(allocation)
                or int(row.maximum_setting_shots) != max(allocation)
            ):
                raise ValueError("finite-shot allocation extrema do not match allocation_json")
            if mode == "equal" and max(allocation) - min(allocation) > 1:
                raise ValueError("equal allocation is not evenly split")
            if int(row.mc_repeats) != 16:
                raise ValueError("finite-shot rows must have 16 MC repeats")
        else:
            raise ValueError("finite-shot allocation mode differs from the v2 profile")
        normalized_keys.append(
            (
                str(row.root_id),
                int(row.teacher_n_qubits),
                budget,
                mode,
                normalized_total,
            )
        )

    expected_keys: set[tuple[object, ...]] = set()
    for root, n_qubits, budget in product(
        SHOT_ROOTS, SHOT_TEACHER_N, SHOT_BUDGETS
    ):
        expected_keys.add((root, n_qubits, budget, "noiseless", None))
        for mode, total in product(("equal", "optimal"), SHOT_TOTALS):
            expected_keys.add((root, n_qubits, budget, mode, total))
    if len(normalized_keys) != len(set(normalized_keys)):
        raise ValueError("finite-shot table has duplicate Cartesian keys")
    observed_keys = set(normalized_keys)
    if observed_keys != expected_keys or len(results) != len(expected_keys):
        missing = sorted(expected_keys - observed_keys, key=repr)[:3]
        extra = sorted(observed_keys - expected_keys, key=repr)[:3]
        raise ValueError(
            "finite-shot Cartesian keys differ from the exact v2 profile; "
            f"missing={missing!r}, extra={extra!r}"
        )
    nonnegative = (
        "validation_noiseless_mse",
        "validation_expected_shot_penalty_mse",
        "validation_expected_objective_mse",
        "test_target_variance",
        "test_noiseless_mse",
        "test_noiseless_nrmse",
        "test_expected_shot_penalty_mse",
        "test_expected_total_mse",
        "test_expected_total_nrmse",
    )
    if any(bool((results[column] < 0).any()) for column in nonnegative):
        raise ValueError("finite-shot table contains a negative error or variance")
    if bool((results["test_target_variance"] <= 0).any()):
        raise ValueError("finite-shot test target variance must be positive")
    if not np.allclose(
        results["validation_expected_objective_mse"],
        results["validation_noiseless_mse"]
        + results["validation_expected_shot_penalty_mse"],
        atol=2e-12,
        rtol=0.0,
    ):
        raise ValueError("finite-shot validation objective identity failed")
    if not np.allclose(
        results["test_expected_total_mse"],
        results["test_noiseless_mse"] + results["test_expected_shot_penalty_mse"],
        atol=2e-12,
        rtol=0.0,
    ):
        raise ValueError("finite-shot test MSE identity failed")


def derive_measurement_visibility(results: pd.DataFrame) -> pd.DataFrame:
    """Derive the exact Pauli-Parseval measured-memory visibility fraction."""

    denominator = (
        np.power(2.0, results["teacher_n_qubits"].to_numpy(dtype=float))
        * results["final_state_hs_mse"].to_numpy(dtype=float)
    )
    if bool(np.any(~np.isfinite(denominator))) or bool(np.any(denominator <= 0)):
        raise ValueError("measurement-visibility denominator must be finite and positive")
    fraction = (
        results["n_outputs"].to_numpy(dtype=float)
        * results["history_mse"].to_numpy(dtype=float)
        / denominator
    )
    if bool(np.any(~np.isfinite(fraction))):
        raise ValueError("measurement visibility is nonfinite")
    if bool(np.any(fraction < -VISIBILITY_TOLERANCE)) or bool(
        np.any(fraction > 1.0 + VISIBILITY_TOLERANCE)
    ):
        raise RuntimeError("measurement visibility violates the Pauli-Parseval interval")
    output = results.loc[:, list(OBSERVABLE_KEY_COLUMNS)].copy()
    output["n_outputs"] = results["n_outputs"].to_numpy(dtype=int)
    output["history_mse"] = results["history_mse"].to_numpy(dtype=float)
    output["final_state_hs_mse"] = results["final_state_hs_mse"].to_numpy(
        dtype=float
    )
    output["f_vis"] = fraction
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_median(values: Iterable[float], *, seed: int) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a nonempty finite vector")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(array, size=(20_000, len(array)), replace=True)
    medians = np.median(draws, axis=1)
    return {
        "n_roots": int(len(array)),
        "median": float(np.median(array)),
        "ci95_low": float(np.quantile(medians, 0.025)),
        "ci95_high": float(np.quantile(medians, 0.975)),
        "positive_roots": int(np.sum(array > 0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def select_budget_on_validation(finite: pd.DataFrame) -> pd.DataFrame:
    """Select B without touching test performance.

    Ridge and allocation are already selected inside each B cell.  This final
    resource choice uses only the stored validation objective; test NRMSE is
    joined by retaining the selected row.
    """

    keys = ["root_id", "teacher_n_qubits", "allocation_mode", "total_shots"]
    ordered = finite.sort_values(
        keys + ["snapshot_budget"], kind="mergesort"
    )
    selected = ordered.loc[
        ordered.groupby(keys, sort=True)["validation_expected_objective_mse"].idxmin()
    ].copy()
    return selected.sort_values(keys, kind="mergesort").reset_index(drop=True)


def analyze_observable(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    validate_observable_v2_profile(results)
    required = {
        "root_id",
        "teacher_n_qubits",
        "dissipator_family",
        "reference_local_loss_gamma",
        "observable_panel",
        "window",
        "target_variance",
        "history_mse",
        "history_nrmse",
        "transfer_predicted_history_mse",
        "max_transfer_prediction_error",
        "reference_jump_strength",
        "matched_jump_strength",
        "tracezero_bound_slack",
        "sample_span_bound_slack",
        "n_outputs",
        "final_state_hs_mse",
    }
    if missing := required - set(results):
        raise ValueError(f"observable table missing columns: {sorted(missing)}")
    if len(results) != 3_888 or results["root_id"].nunique() != 12:
        raise ValueError("observable table is not the 12-root 3,888-row profile")
    if set(results["teacher_n_qubits"]) != {2, 3, 4}:
        raise ValueError("unexpected teacher-size axis")
    if set(results["dissipator_family"]) != {
        "local_loss",
        "dephasing",
        "collective_loss",
    }:
        raise ValueError("unexpected dissipator axis")

    max_direct_mse_error = float(
        np.max(
            np.abs(
                results["history_mse"]
                - results["transfer_predicted_history_mse"]
            )
        )
    )
    invariants = {
        "maximum_recorded_transfer_output_error": float(
            results["max_transfer_prediction_error"].max()
        ),
        "maximum_direct_vs_transfer_history_mse_error": max_direct_mse_error,
        "maximum_jump_strength_mismatch": float(
            np.max(
                np.abs(
                    results["reference_jump_strength"]
                    - results["matched_jump_strength"]
                )
            )
        ),
        "minimum_tracezero_bound_slack": float(
            results["tracezero_bound_slack"].min()
        ),
        "minimum_sample_span_bound_slack": float(
            results["sample_span_bound_slack"].min()
        ),
    }
    if invariants["maximum_recorded_transfer_output_error"] > 2e-12:
        raise RuntimeError("transfer-output identity failed")
    if max_direct_mse_error > 2e-12:
        raise RuntimeError("transfer-risk identity failed")
    if invariants["maximum_jump_strength_mismatch"] > 1e-12:
        raise RuntimeError("jump-strength matching failed")
    if (
        invariants["minimum_tracezero_bound_slack"] < -2e-12
        or invariants["minimum_sample_span_bound_slack"] < -2e-12
    ):
        raise RuntimeError("observable upper bound failed")

    visibility_rows = derive_measurement_visibility(results)
    liveness_floor = 1e-8
    results = results.copy()
    results["f_vis"] = visibility_rows["f_vis"].to_numpy(dtype=float)
    results["live"] = results["target_variance"] >= liveness_floor
    cell_keys = [
        "root_id",
        "teacher_n_qubits",
        "dissipator_family",
        "reference_local_loss_gamma",
        "observable_panel",
    ]
    paired = results.pivot(
        index=cell_keys,
        columns="window",
        values=["history_nrmse", "target_variance"],
    )
    paired = paired[
        (paired[("target_variance", 1)] >= liveness_floor)
        & (paired[("target_variance", 8)] >= liveness_floor)
    ].copy()
    paired["gain_d1_to_d8"] = (
        1.0
        - paired[("history_nrmse", 8)] / paired[("history_nrmse", 1)]
    )
    root_rows: list[dict[str, float | int | str]] = []
    for (root_id, family), group in paired.groupby(
        level=["root_id", "dissipator_family"], sort=True
    ):
        root_rows.append(
            {
                "root_id": str(root_id),
                "dissipator_family": str(family),
                "paired_live_cells": int(len(group)),
                "median_history_nrmse_d1": float(
                    np.median(group[("history_nrmse", 1)])
                ),
                "median_history_nrmse_d8": float(
                    np.median(group[("history_nrmse", 8)])
                ),
                "gain_d1_to_d8": float(np.median(group["gain_d1_to_d8"])),
            }
        )
    root_endpoints = pd.DataFrame(root_rows)
    effects: dict[str, object] = {}
    for offset, (family, group) in enumerate(
        root_endpoints.groupby("dissipator_family", sort=True)
    ):
        effects[str(family)] = {
            "median_history_nrmse_d1": float(
                np.median(group["median_history_nrmse_d1"])
            ),
            "median_history_nrmse_d8": float(
                np.median(group["median_history_nrmse_d8"])
            ),
            "fractional_reduction": bootstrap_median(
                group["gain_d1_to_d8"], seed=28_082_600 + offset
            ),
            "paired_live_cells_total": int(group["paired_live_cells"].sum()),
        }

    # A between-family sensitivity must compare the same root/size/rate/panel
    # cells.  The within-family endpoint summaries above condition on each
    # family's own live cells and therefore are not direct family contrasts.
    common_index = [
        "root_id",
        "teacher_n_qubits",
        "reference_local_loss_gamma",
        "observable_panel",
    ]
    common = results[results["window"].isin((1, 8))].pivot(
        index=common_index,
        columns=["dissipator_family", "window"],
        values=["history_nrmse", "target_variance"],
    )
    common_live = np.ones(len(common), dtype=bool)
    for family, window in product(OBSERVABLE_FAMILIES, (1, 8)):
        common_live &= (
            common[("target_variance", family, window)].to_numpy(dtype=float)
            >= liveness_floor
        )
    common = common.loc[common_live].copy()
    common_root_rows: list[dict[str, float | int | str]] = []
    for family in OBSERVABLE_FAMILIES:
        gain = 1.0 - (
            common[("history_nrmse", family, 8)]
            / common[("history_nrmse", family, 1)]
        )
        gain = gain.rename("gain_d1_to_d8").reset_index()
        for root_id, group in gain.groupby("root_id", sort=True):
            common_root_rows.append(
                {
                    "root_id": str(root_id),
                    "dissipator_family": str(family),
                    "common_live_cells": int(len(group)),
                    "common_live_gain_d1_to_d8": float(
                        np.median(group["gain_d1_to_d8"])
                    ),
                }
            )
    common_root = pd.DataFrame(common_root_rows)
    root_endpoints = root_endpoints.merge(
        common_root,
        on=["root_id", "dissipator_family"],
        validate="one_to_one",
    )
    common_effects: dict[str, object] = {}
    for offset, (family, group) in enumerate(
        common_root.groupby("dissipator_family", sort=True)
    ):
        common_effects[str(family)] = {
            "fractional_reduction": bootstrap_median(
                group["common_live_gain_d1_to_d8"],
                seed=28_082_680 + offset,
            ),
            "common_live_cells_total": int(group["common_live_cells"].sum()),
        }

    high_loss = results[
        (results["reference_local_loss_gamma"] == 4.0)
        & (results["window"] == 8)
        & results["live"]
    ]
    high_loss_cells: list[dict[str, float | int | str]] = []
    for key, group in high_loss.groupby(
        ["dissipator_family", "teacher_n_qubits"], sort=True
    ):
        root_values = group.groupby("root_id")["history_nrmse"].median()
        high_loss_cells.append(
            {
                "dissipator_family": str(key[0]),
                "teacher_n_qubits": int(key[1]),
                "n_roots": int(len(root_values)),
                "median_history_nrmse": float(np.median(root_values)),
                "minimum_history_nrmse": float(np.min(root_values)),
                "maximum_history_nrmse": float(np.max(root_values)),
            }
        )
    liveness: list[dict[str, float | int | str]] = []
    for family, group in results.groupby("dissipator_family", sort=True):
        liveness.append(
            {
                "dissipator_family": str(family),
                "live_rows": int(group["live"].sum()),
                "total_rows": int(len(group)),
                "live_fraction": float(group["live"].mean()),
            }
        )
    visibility_summaries: list[dict[str, float | int | str]] = []
    visibility_focus = results[
        (results["reference_local_loss_gamma"] == 4.0)
        & (results["window"] == 8)
    ]
    visibility_keys = [
        "dissipator_family",
        "teacher_n_qubits",
        "observable_panel",
    ]
    for offset, (key, group) in enumerate(
        visibility_focus.groupby(visibility_keys, sort=True)
    ):
        if group["root_id"].nunique() != len(OBSERVABLE_ROOTS):
            raise RuntimeError("measurement-visibility root block is incomplete")
        effect = bootstrap_median(
            group["f_vis"], seed=28_082_660 + offset
        )
        visibility_summaries.append(
            {
                "dissipator_family": str(key[0]),
                "teacher_n_qubits": int(key[1]),
                "observable_panel": str(key[2]),
                **effect,
            }
        )
    summary = {
        "counts": {
            "rows": int(len(results)),
            "roots": int(results["root_id"].nunique()),
            "teacher_sizes": 3,
            "dissipators": 3,
            "rates": 3,
            "windows": 4,
            "observable_panels": 3,
        },
        "invariants": invariants,
        "liveness": {
            "target_variance_floor": liveness_floor,
            "by_dissipator": liveness,
            "interpretation": (
                "NRMSE endpoint effects use only cells live at both d=1 and d=8; "
                "dephasing frequently suppresses total output variance below the floor"
            ),
        },
        "root_block_effects": effects,
        "common_live_support_sensitivity": {
            "cell_definition": (
                "same root, teacher size, reference rate, and observable panel; "
                "target variance is live at d=1 and d=8 in all three dissipators"
            ),
            "effects": common_effects,
            "interpretation": (
                "within-family endpoint reductions condition on different live-cell "
                "mixtures and are not direct between-family effect estimates"
            ),
        },
        "gamma4_d8_cells": high_loss_cells,
        "measurement_visibility": {
            "formula": (
                "n_outputs * history_mse / "
                "(2**teacher_n_qubits * final_state_hs_mse)"
            ),
            "parseval_tolerance": VISIBILITY_TOLERANCE,
            "rows": int(len(results)),
            "minimum": float(results["f_vis"].min()),
            "maximum": float(results["f_vis"].max()),
            "gamma4_d8_root_summaries": visibility_summaries,
        },
    }
    return root_endpoints, summary


def analyze_shots(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    validate_shot_v2_profile(results)
    if len(results) != 648 or results["root_id"].nunique() != 12:
        raise ValueError("finite-shot table is not the 12-root 648-row profile")
    finite = results[results["allocation_mode"].ne("noiseless")].copy()
    noiseless = results[results["allocation_mode"].eq("noiseless")].copy()

    noiseless_wide = noiseless.pivot(
        index=["root_id", "teacher_n_qubits"],
        columns="snapshot_budget",
        values="test_noiseless_nrmse",
    )
    noiseless_effect = 1.0 - noiseless_wide[4] / noiseless_wide[1]
    noiseless_root = noiseless_effect.groupby("root_id").median()

    finite_wide = finite.pivot(
        index=["root_id", "teacher_n_qubits", "allocation_mode", "total_shots"],
        columns="snapshot_budget",
        values="test_expected_total_nrmse",
    )
    finite_wide["gain_B72_vs_B18"] = 1.0 - finite_wide[4] / finite_wide[1]
    finite_wide["gain_B72_vs_B36"] = 1.0 - finite_wide[4] / finite_wide[2]

    allocation_wide = finite.pivot(
        index=["root_id", "teacher_n_qubits", "snapshot_budget", "total_shots"],
        columns="allocation_mode",
        values="test_expected_total_nrmse",
    )
    allocation_wide["gain_optimal_vs_equal"] = (
        1.0 - allocation_wide["optimal"] / allocation_wide["equal"]
    )

    effects: dict[str, object] = {
        "noiseless_B72_vs_B18": bootstrap_median(
            noiseless_root, seed=28_082_610
        ),
        "B72_vs_B18_by_allocation_and_Q": {},
        "B72_vs_B36_by_allocation_and_Q": {},
        "optimal_vs_equal_by_Q": {},
    }
    for offset, ((mode, total), group) in enumerate(
        finite_wide.groupby(level=["allocation_mode", "total_shots"], sort=True)
    ):
        root_effect = group["gain_B72_vs_B18"].groupby("root_id").median()
        key = f"{mode}_Q{int(total)}"
        effects["B72_vs_B18_by_allocation_and_Q"][key] = bootstrap_median(
            root_effect, seed=28_082_620 + offset
        )
        root_effect_36 = group["gain_B72_vs_B36"].groupby("root_id").median()
        effects["B72_vs_B36_by_allocation_and_Q"][key] = bootstrap_median(
            root_effect_36, seed=28_082_630 + offset
        )
    for offset, (total, group) in enumerate(
        allocation_wide.groupby(level="total_shots", sort=True)
    ):
        root_effect = group["gain_optimal_vs_equal"].groupby("root_id").median()
        effects["optimal_vs_equal_by_Q"][f"Q{int(total)}"] = bootstrap_median(
            root_effect, seed=28_082_640 + offset
        )

    surface: list[dict[str, float | int | str]] = []
    for key, group in finite.groupby(
        ["snapshot_budget", "allocation_mode", "total_shots"], sort=True
    ):
        surface.append(
            {
                "snapshot_budget": int(key[0]),
                "feature_budget": int(key[0]) * 18,
                "allocation_mode": str(key[1]),
                "total_shots": int(key[2]),
                "median_expected_test_nrmse": float(
                    np.median(group["test_expected_total_nrmse"])
                ),
            }
        )

    validation_selected = select_budget_on_validation(finite)
    selected_counts: list[dict[str, int | str]] = []
    for key, count in validation_selected.groupby(
        ["allocation_mode", "total_shots", "snapshot_budget"], sort=True
    ).size().items():
        selected_counts.append(
            {
                "allocation_mode": str(key[0]),
                "total_shots": int(key[1]),
                "snapshot_budget": int(key[2]),
                "feature_budget": int(key[2]) * 18,
                "root_teacher_cells": int(count),
            }
        )

    positive = finite["test_expected_shot_penalty_mse"] > 0
    ratios = (
        finite.loc[positive, "mc_frozen_penalty_mean_mse"]
        / finite.loc[positive, "test_expected_shot_penalty_mse"]
    )
    summary = {
        "counts": {
            "rows": int(len(results)),
            "roots": int(results["root_id"].nunique()),
            "root_teacher_cells": int(
                results[["root_id", "teacher_n_qubits"]].drop_duplicates().shape[0]
            ),
        },
        "noiseless_median_test_nrmse": {
            f"B{int(snapshot) * 18}": float(np.median(group["test_noiseless_nrmse"]))
            for snapshot, group in noiseless.groupby("snapshot_budget", sort=True)
        },
        "root_block_effects": effects,
        "finite_shot_surface": surface,
        "validation_selected_budget_counts": selected_counts,
        "validation_selected_test_surface": [
            {
                "allocation_mode": str(key[0]),
                "total_shots": int(key[1]),
                "n_root_teacher_cells": int(len(group)),
                "median_heldout_expected_nrmse": float(
                    np.median(group["test_expected_total_nrmse"])
                ),
            }
            for key, group in validation_selected.groupby(
                ["allocation_mode", "total_shots"], sort=True
            )
        ],
        "monte_carlo_to_exact_penalty_ratio": {
            "median": float(np.median(ratios)),
            "minimum": float(np.min(ratios)),
            "maximum": float(np.max(ratios)),
            "repeats_per_cell": int(finite["mc_repeats"].iloc[0]),
        },
    }
    root_effects = finite_wide.reset_index()
    return root_effects, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observable", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    observable = pd.read_csv(args.observable)
    shots = pd.read_csv(args.shots)
    observable_roots, observable_summary = analyze_observable(observable)
    observable_visibility = derive_measurement_visibility(observable)
    shot_roots, shot_summary = analyze_shots(shots)

    observable_path = output / "observable_root_effects.csv"
    visibility_path = output / "observable_visibility_rows.csv"
    shot_path = output / "finite_shot_root_effects.csv"
    observable_roots.to_csv(observable_path, index=False, float_format="%.17g")
    observable_visibility.to_csv(
        visibility_path, index=False, float_format="%.17g"
    )
    shot_roots.to_csv(shot_path, index=False, float_format="%.17g")
    payload = {
        "schema_version": 1,
        "status": "EXPLORATORY_POST_CONFIRMATORY",
        "inferential_unit": "root_block",
        "bootstrap": {
            "replicates": 20_000,
            "interval": "percentile_95",
            "deterministic_seeds": True,
        },
        "source_artifacts": {
            str(args.observable): sha256(args.observable),
            str(args.shots): sha256(args.shots),
        },
        "observable_memory": observable_summary,
        "finite_shots": shot_summary,
    }
    summary_path = output / "pra_extension_analysis.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = {
        path.name: sha256(path)
        for path in (observable_path, visibility_path, shot_path, summary_path)
    }
    manifest_path = output / "artifact_sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output), "sha256": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
