#!/usr/bin/env python3
"""Deterministic finite-K calibration audit for the frozen suffix experiment.

The registered suffix decomposition deliberately uses the divisor ``K`` so that
its realized-panel Pythagorean identity is exact.  Under the additional model
that the K prefix branches within each suffix group are conditionally iid, its
history numerator estimates only ``(K - 1) / K`` of the population conditional
variance.  This companion analysis applies the corresponding ``K / (K - 1)``
factor to the history *numerator* while retaining the registered target variance.

The calibrated quantity is a post-hoc population-prefix sensitivity analysis.
It never replaces the registered empirical decomposition or its decision gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUFFIX = (
    REPO_ROOT
    / "results_canonical_confirmatory_v1"
    / "suffix_decomposition.csv"
)
DEFAULT_GATE_REPORT = (
    REPO_ROOT
    / "results_canonical_confirmatory_v1"
    / "canonical_gate_report.json"
)
DEFAULT_PHASE_SOURCE = (
    REPO_ROOT
    / "paper"
    / "figures"
    / "canonical_emulation"
    / "fig04_regime_map_source.csv"
)
OUTPUT_NAME = "finite_k_history_calibration.json"

EXPECTED_SHA256 = {
    "suffix_decomposition.csv": (
        "94ec3e840ac1866e675dbb796aa7b72010600c2408ba7064455c686dac9d88b1"
    ),
    "canonical_gate_report.json": (
        "69ec32b96492c92d4b193f6b2cc22afc3598fc7e7115ba497e3d0a8726000a06"
    ),
    "fig04_regime_map_source.csv": (
        "a33a854e271f3bf38f97acb5d731800afc52239dc7921474b542bb7131d61e9f"
    ),
}

ROOTS = tuple(f"confirm{index:02d}" for index in range(1, 13))
TEACHER_NS = (2, 3)
GAMMAS = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)
WINDOWS = (1, 2, 4, 6)
SNAPSHOT_BUDGETS = (1, 2, 4)
FEATURE_BUDGETS = (18, 36, 72)
REGIME_ORDER = (
    "DEGENERATE_TARGET",
    "HISTORY_IMPOSSIBLE",
    "EMULATED",
    "REPRESENTATION_LIMITED",
    "UNRESOLVED",
)
MODAL_ORDER = (*REGIME_ORDER, "MIXED")
EXPECTED_MODEL_COUNTS = {
    "classical_tanh": 1_728,
    "constant": 576,
    "linear_delay": 576,
    "poly2_delay": 576,
    "ptd_parallel": 1_728,
    "rr_same_identity": 576,
}
REQUIRED_SUFFIX_COLUMNS = {
    "root_id",
    "system_id",
    "teacher_n_qubits",
    "teacher_dissipator",
    "teacher_gamma",
    "window",
    "student_n_qubits",
    "model",
    "snapshot_budget",
    "feature_budget",
    "history_mse",
    "representation_mse",
    "total_mse",
    "target_variance",
    "identity_error",
    "n_suffix_groups",
    "branches_per_group",
    "n_outputs",
    "history_nrmse",
    "representation_nrmse",
    "total_nrmse",
    "oracle_nrmse",
    "regime",
}
REQUIRED_PHASE_COLUMNS = {
    "teacher_n_qubits",
    "teacher_gamma",
    "window",
    "feature_budget",
    "modal_regime",
    "agreement_n",
    "n_roots",
    "regimes_present",
    "provenance_decision",
    "provenance_input_canonical_gate_report_json_sha256",
    "provenance_input_suffix_decomposition_csv_sha256",
}
SUFFIX_KEY = (
    "root_id",
    "system_id",
    "teacher_n_qubits",
    "teacher_gamma",
    "window",
    "model",
    "snapshot_budget",
    "feature_budget",
)
PHASE_KEY = (
    "teacher_n_qubits",
    "teacher_gamma",
    "window",
    "feature_budget",
)


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load one JSON object, rejecting duplicate keys and nonstandard numbers."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(left: float, right: float, *, atol: float = 2e-12) -> bool:
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=0.0))


def _maximum_absolute(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _regime_counts(values: Iterable[str], *, modal: bool = False) -> dict[str, int]:
    series = pd.Series(list(values), dtype="object")
    observed = series.value_counts().to_dict()
    order = MODAL_ORDER if modal else REGIME_ORDER
    unknown = sorted(set(observed).difference(order))
    _require(not unknown, f"unknown regime labels: {unknown}")
    return {name: int(observed.get(name, 0)) for name in order}


def classify_regime(
    *,
    history_nrmse: float,
    total_nrmse: float,
    oracle_nrmse: float,
    target_variance: float,
    tolerance: float,
    min_target_variance: float,
) -> str:
    """Mirror the frozen regime decision order without mutating its source data."""

    values = np.asarray(
        [
            history_nrmse,
            total_nrmse,
            oracle_nrmse,
            target_variance,
            tolerance,
            min_target_variance,
        ],
        dtype=float,
    )
    _require(bool(np.isfinite(values).all()), "regime inputs must be finite")
    _require(tolerance >= 0.0, "NRMSE tolerance must be nonnegative")
    _require(
        min_target_variance >= 0.0,
        "minimum target variance must be nonnegative",
    )
    if target_variance < min_target_variance:
        return "DEGENERATE_TARGET"
    if history_nrmse > tolerance:
        return "HISTORY_IMPOSSIBLE"
    if total_nrmse <= tolerance:
        return "EMULATED"
    if oracle_nrmse <= tolerance:
        return "REPRESENTATION_LIMITED"
    return "UNRESOLVED"


def _validate_gate_report(gate: Mapping[str, Any], suffix_sha256: str) -> None:
    required = {
        "schema_version",
        "study",
        "profile",
        "confirmatory",
        "decision",
        "expected_roots",
        "minimum_successful_roots",
        "statistical_unit",
        "frozen_decision_config",
        "primary_gates",
        "secondary_p6_forgetting_scaling",
        "evaluation_provenance",
    }
    _require(required.issubset(gate), "gate report is missing required fields")
    _require(gate["schema_version"] == 1, "unexpected gate-report schema")
    _require(
        gate["study"] == "canonical_qrc_to_parallel_qelm_emulation",
        "unexpected gate-report study",
    )
    _require(gate["profile"] == "confirmatory", "gate report is not confirmatory")
    _require(gate["confirmatory"] is True, "confirmatory flag is not true")
    _require(gate["expected_roots"] == 12, "gate report does not expect 12 roots")
    _require(
        gate["minimum_successful_roots"] == 10,
        "unexpected minimum successful roots",
    )
    _require(gate["statistical_unit"] == "root_block", "wrong statistical unit")
    _require(
        gate["decision"] == "CONFIRMS_FINITE_RESOURCE_FRONTIER",
        "unexpected frozen decision",
    )

    config = gate["frozen_decision_config"]
    expected_config = {
        "g4_gammas": [2.0, 4.0],
        "g5_gamma": 4.0,
        "min_best_classical_gain": 0.1,
        "min_feature_gain": 0.2,
        "min_g2_live_systems_per_family": 5,
        "min_root_successes": 10,
        "min_scaling_r_squared": 0.85,
        "min_spearman": 0.8,
        "min_target_variance": 1e-8,
        "min_window_gain": 0.2,
        "roots_total": 12,
        "tolerance_nrmse": 0.1,
    }
    for key, expected in expected_config.items():
        _require(key in config, f"frozen decision config missing {key!r}")
        observed = config[key]
        if isinstance(expected, float):
            _require(_close(observed, expected), f"unexpected frozen value for {key}")
        else:
            _require(observed == expected, f"unexpected frozen value for {key}")

    primary = gate["primary_gates"]
    _require(
        set(primary) == {f"G{index}" for index in range(7)},
        "primary-gate set differs from G0--G6",
    )
    _require(
        all(primary[name].get("passed") is True for name in primary),
        "not every frozen primary gate passed",
    )
    _require(primary["G2"]["successful_roots"] == 12, "unexpected G2 result")
    _require(
        _close(primary["G2"]["median_root_median_spearman"], 1.0),
        "unexpected G2 median Spearman",
    )
    for teacher_n in TEACHER_NS:
        family = primary["G3"]["families"][str(teacher_n)]
        _require(family["successful_roots"] == 12, "unexpected G3 result")
        _require(
            len(family["root_success"]) == 12
            and all(family["root_success"].values()),
            "unexpected G3 root pattern",
        )
    _require(primary["G4"]["positive_roots"] == 12, "unexpected G4 result")
    _require(
        _close(primary["G4"]["median_gain"], 0.784512834334567),
        "unexpected G4 median gain",
    )
    _require(
        len(primary["G4"]["history_transition_cells"]) == 48
        and all(primary["G4"]["history_transition_cells"].values()),
        "unexpected G4 transition pattern",
    )
    _require(
        len(primary["G4"]["history_transition_root_success"]) == 12
        and all(primary["G4"]["history_transition_root_success"].values()),
        "unexpected G4 root-transition pattern",
    )
    _require(primary["G5"]["positive_roots"] == 12, "unexpected G5 result")
    _require(
        _close(primary["G5"]["median_gain"], 0.6726414464648288),
        "unexpected G5 median gain",
    )
    _require(
        primary["G5"]["history_feasible_root_family_cells"] == 24
        and len(primary["G5"]["history_feasibility_by_root_family"]) == 24
        and all(primary["G5"]["history_feasibility_by_root_family"].values()),
        "unexpected G5 feasibility pattern",
    )
    expected_g6 = {2: 0.661282098534065, 3: 0.6884823496693037}
    for teacher_n, expected in expected_g6.items():
        family = primary["G6"]["families"][str(teacher_n)]
        _require(family["passed"] is True, "unexpected G6 family result")
        _require(
            _close(family["g5"]["median_gain"], expected),
            "unexpected G6 family median",
        )

    provenance = gate["evaluation_provenance"]["input_artifact_sha256"]
    _require(
        provenance["suffix_decomposition.csv"] == suffix_sha256,
        "gate report does not authenticate the supplied suffix table",
    )


def validate_suffix_frame(
    suffix: pd.DataFrame,
    *,
    tolerance: float,
    min_target_variance: float,
) -> dict[str, Any]:
    """Fail closed unless the table is the complete frozen suffix profile."""

    missing = sorted(REQUIRED_SUFFIX_COLUMNS.difference(suffix.columns))
    _require(not missing, f"suffix table missing columns: {missing}")
    _require(len(suffix) == 5_760, "suffix table must contain exactly 5,760 rows")
    _require(
        not bool(suffix.duplicated(list(SUFFIX_KEY)).any()),
        "suffix logical key is duplicated",
    )

    numeric_columns = (
        "teacher_n_qubits",
        "teacher_gamma",
        "window",
        "student_n_qubits",
        "feature_budget",
        "history_mse",
        "representation_mse",
        "total_mse",
        "target_variance",
        "identity_error",
        "n_suffix_groups",
        "branches_per_group",
        "n_outputs",
        "history_nrmse",
        "representation_nrmse",
        "total_nrmse",
        "oracle_nrmse",
    )
    converted = suffix.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    _require(
        bool(np.isfinite(converted.to_numpy(dtype=float)).all()),
        "suffix numeric fields must be finite",
    )
    nonnegative = (
        "history_mse",
        "representation_mse",
        "total_mse",
        "target_variance",
        "history_nrmse",
        "representation_nrmse",
        "total_nrmse",
        "oracle_nrmse",
    )
    _require(
        not bool((converted.loc[:, nonnegative] < 0.0).any().any()),
        "suffix risks must be nonnegative",
    )
    _require(
        bool((converted["target_variance"] > 0.0).all()),
        "frozen target variances must be positive",
    )

    _require(tuple(sorted(suffix["root_id"].astype(str).unique())) == ROOTS, "root grid differs")
    _require(
        tuple(sorted(converted["teacher_n_qubits"].astype(int).unique()))
        == TEACHER_NS,
        "teacher-family grid differs",
    )
    _require(
        tuple(sorted(converted["teacher_gamma"].astype(float).unique())) == GAMMAS,
        "gamma grid differs",
    )
    _require(
        tuple(sorted(converted["window"].astype(int).unique())) == WINDOWS,
        "window grid differs",
    )
    _require(
        set(suffix["teacher_dissipator"].astype(str)) == {"local_loss"},
        "unexpected teacher dissipator",
    )
    _require(
        set(converted["student_n_qubits"].astype(int)) == {6},
        "unexpected student register size",
    )
    _require(
        set(converted["n_suffix_groups"].astype(int)) == {32},
        "suffix-group count is not uniformly 32",
    )
    _require(
        set(converted["branches_per_group"].astype(int)) == {4},
        "branches-per-group count is not uniformly four",
    )
    model_counts = {
        str(key): int(value)
        for key, value in suffix["model"].astype(str).value_counts().items()
    }
    _require(model_counts == EXPECTED_MODEL_COUNTS, "model row counts differ")
    _require(suffix["system_id"].nunique() == 144, "system count differs")

    ptd = suffix[suffix["model"].astype(str).eq("ptd_parallel")].copy()
    _require(
        not bool(ptd["snapshot_budget"].isna().any()),
        "PTD snapshot budget contains nulls",
    )
    observed_ptd = {
        (
            str(row.root_id),
            int(row.teacher_n_qubits),
            float(row.teacher_gamma),
            int(row.window),
            int(row.snapshot_budget),
        )
        for row in ptd.itertuples(index=False)
    }
    expected_ptd = set(product(ROOTS, TEACHER_NS, GAMMAS, WINDOWS, SNAPSHOT_BUDGETS))
    _require(observed_ptd == expected_ptd, "PTD Cartesian profile differs")
    _require(
        bool(
            (
                ptd["feature_budget"].to_numpy(dtype=int)
                == 18 * ptd["snapshot_budget"].to_numpy(dtype=int)
            ).all()
        ),
        "PTD feature-budget mapping differs",
    )

    rr = suffix[suffix["model"].astype(str).eq("rr_same_identity")].copy()
    observed_rr = {
        (
            str(row.root_id),
            int(row.teacher_n_qubits),
            float(row.teacher_gamma),
            int(row.window),
        )
        for row in rr.itertuples(index=False)
    }
    expected_rr = set(product(ROOTS, TEACHER_NS, GAMMAS, WINDOWS))
    _require(observed_rr == expected_rr, "reset-replay Cartesian profile differs")

    invariant_columns = (
        "history_mse",
        "history_nrmse",
        "target_variance",
        "oracle_nrmse",
        "n_suffix_groups",
        "branches_per_group",
        "n_outputs",
    )
    invariant_group = [
        "root_id",
        "system_id",
        "teacher_n_qubits",
        "teacher_gamma",
        "window",
    ]
    variation = suffix.groupby(invariant_group, sort=True)[list(invariant_columns)].nunique(dropna=False)
    _require(
        bool((variation.to_numpy(dtype=int) == 1).all()),
        "history quantities change across student models or budgets",
    )

    empirical_residual = (
        suffix["total_mse"].to_numpy(dtype=float)
        - suffix["history_mse"].to_numpy(dtype=float)
        - suffix["representation_mse"].to_numpy(dtype=float)
    )
    max_identity_error = _maximum_absolute(empirical_residual)
    _require(max_identity_error <= 1e-12, "empirical decomposition identity failed")
    _require(
        _maximum_absolute(suffix["identity_error"].to_numpy(dtype=float)) <= 1e-12,
        "recorded identity-error field exceeds tolerance",
    )
    normalized_checks = {
        "history_nrmse": np.sqrt(suffix["history_mse"] / suffix["target_variance"]),
        "representation_nrmse": np.sqrt(
            suffix["representation_mse"] / suffix["target_variance"]
        ),
        "total_nrmse": np.sqrt(suffix["total_mse"] / suffix["target_variance"]),
    }
    for column, expected in normalized_checks.items():
        max_error = _maximum_absolute(suffix[column].to_numpy(dtype=float) - expected)
        _require(max_error <= 2e-12, f"{column} normalization identity failed")

    expected_regimes = [
        classify_regime(
            history_nrmse=float(row.history_nrmse),
            total_nrmse=float(row.total_nrmse),
            oracle_nrmse=float(row.oracle_nrmse),
            target_variance=float(row.target_variance),
            tolerance=float(tolerance),
            min_target_variance=float(min_target_variance),
        )
        for row in suffix.itertuples(index=False)
    ]
    _require(
        expected_regimes == suffix["regime"].astype(str).tolist(),
        "recorded regimes differ from the frozen decision order",
    )
    return {
        "rows": 5_760,
        "ptd_root_rows": 1_728,
        "roots": 12,
        "systems": 144,
        "suffix_groups_per_panel": 32,
        "branches_per_group": 4,
        "logical_key_duplicates": 0,
        "history_student_invariance": True,
        "max_empirical_decomposition_error": max_identity_error,
        "normalized_metric_identities": True,
        "recorded_regimes_match_frozen_classifier": True,
    }


def _phase_table(ptd: pd.DataFrame, regime_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, frame in ptd.groupby(list(PHASE_KEY), sort=True):
        counts = frame[regime_column].astype(str).value_counts().to_dict()
        unknown = sorted(set(counts).difference(REGIME_ORDER))
        _require(not unknown, f"unknown phase-map labels: {unknown}")
        best = max(counts.values())
        winners = [name for name in REGIME_ORDER if counts.get(name, 0) == best]
        modal = winners[0] if len(winners) == 1 else "MIXED"
        rows.append(
            {
                **dict(zip(PHASE_KEY, keys, strict=True)),
                "modal_regime": modal,
                "agreement_n": int(best),
                "n_roots": int(len(frame)),
                "regimes_present": ";".join(
                    f"{name}:{counts[name]}" for name in REGIME_ORDER if name in counts
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(list(PHASE_KEY), kind="mergesort").reset_index(drop=True)


def validate_phase_source(
    phase: pd.DataFrame,
    registered_phase: pd.DataFrame,
    *,
    suffix_sha256: str,
    gate_sha256: str,
    decision: str,
) -> dict[str, Any]:
    """Authenticate and exactly reproduce the frozen phase-map source table."""

    missing = sorted(REQUIRED_PHASE_COLUMNS.difference(phase.columns))
    _require(not missing, f"phase source missing columns: {missing}")
    _require(len(phase) == 144, "phase source must contain exactly 144 cells")
    _require(
        not bool(phase.duplicated(list(PHASE_KEY)).any()),
        "phase-map key is duplicated",
    )
    observed_grid = {
        (
            int(row.teacher_n_qubits),
            float(row.teacher_gamma),
            int(row.window),
            int(row.feature_budget),
        )
        for row in phase.itertuples(index=False)
    }
    expected_grid = set(product(TEACHER_NS, GAMMAS, WINDOWS, FEATURE_BUDGETS))
    _require(observed_grid == expected_grid, "phase-map Cartesian profile differs")

    source = phase.sort_values(list(PHASE_KEY), kind="mergesort").reset_index(drop=True)
    for column in PHASE_KEY:
        _require(
            np.array_equal(
                source[column].to_numpy(), registered_phase[column].to_numpy()
            ),
            f"phase-map key column {column!r} differs",
        )
    for column in ("modal_regime", "regimes_present"):
        _require(
            source[column].astype(str).tolist()
            == registered_phase[column].astype(str).tolist(),
            f"phase-map {column!r} differs from suffix rows",
        )
    for column in ("agreement_n", "n_roots"):
        _require(
            np.array_equal(
                source[column].to_numpy(dtype=int),
                registered_phase[column].to_numpy(dtype=int),
            ),
            f"phase-map {column!r} differs from suffix rows",
        )
    _require(bool(source["n_roots"].eq(12).all()), "phase cells are not 12-root blocks")
    _require(
        bool(source["provenance_decision"].astype(str).eq(decision).all()),
        "phase source decision provenance differs",
    )
    _require(
        bool(
            source["provenance_input_suffix_decomposition_csv_sha256"]
            .astype(str)
            .eq(suffix_sha256)
            .all()
        ),
        "phase source does not authenticate the suffix table",
    )
    _require(
        bool(
            source["provenance_input_canonical_gate_report_json_sha256"]
            .astype(str)
            .eq(gate_sha256)
            .all()
        ),
        "phase source does not authenticate the gate report",
    )
    return {
        "cells": 144,
        "roots_per_cell": 12,
        "exactly_reproduced_from_suffix_rows": True,
        "suffix_hash_link_valid": True,
        "gate_hash_link_valid": True,
    }


def _gain(high_resource_error: float, low_resource_error: float) -> float:
    _require(
        np.isfinite(high_resource_error)
        and np.isfinite(low_resource_error)
        and low_resource_error > 0.0,
        "gain inputs must be finite with a positive denominator",
    )
    return float(1.0 - high_resource_error / low_resource_error)


def _g5_gains(
    ptd: pd.DataFrame,
    *,
    metric: str,
    gamma: float,
    window: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    selected = ptd[
        ptd["teacher_gamma"].eq(float(gamma))
        & ptd["window"].eq(int(window))
        & ptd["snapshot_budget"].isin([min(SNAPSHOT_BUDGETS), max(SNAPSHOT_BUDGETS)])
    ]
    by_root_n: dict[str, dict[str, float]] = {root: {} for root in ROOTS}
    for root, teacher_n in product(ROOTS, TEACHER_NS):
        cell = selected[
            selected["root_id"].astype(str).eq(root)
            & selected["teacher_n_qubits"].eq(int(teacher_n))
        ]
        low = cell.loc[cell["snapshot_budget"].eq(min(SNAPSHOT_BUDGETS)), metric]
        high = cell.loc[cell["snapshot_budget"].eq(max(SNAPSHOT_BUDGETS)), metric]
        _require(len(low) == 1 and len(high) == 1, "G5 contrast cell is incomplete")
        by_root_n[root][str(teacher_n)] = _gain(float(high.iloc[0]), float(low.iloc[0]))
    by_root = {
        root: float(np.median(list(by_root_n[root].values()))) for root in ROOTS
    }
    return by_root, by_root_n


def _p6_fits(
    history: pd.DataFrame,
    *,
    metric: str,
    teacher_dt: float,
    minimum_r_squared: float,
) -> dict[str, dict[str, dict[str, float | int | bool]]]:
    output: dict[str, dict[str, dict[str, float | int | bool]]] = {
        str(teacher_n): {} for teacher_n in TEACHER_NS
    }
    expected_cells = set(product(GAMMAS[1:], WINDOWS))
    for teacher_n, root in product(TEACHER_NS, ROOTS):
        rows = history[
            history["teacher_n_qubits"].eq(int(teacher_n))
            & history["root_id"].astype(str).eq(root)
            & history["teacher_gamma"].gt(0.0)
        ].copy()
        observed = {
            (float(row.teacher_gamma), int(row.window))
            for row in rows.itertuples(index=False)
        }
        _require(observed == expected_cells, "P6 regression grid is incomplete")
        errors = rows[metric].to_numpy(dtype=float)
        _require(bool(np.isfinite(errors).all() and (errors > 0.0).all()), "invalid P6 errors")
        x = (
            rows["teacher_gamma"].to_numpy(dtype=float)
            * float(teacher_dt)
            * rows["window"].to_numpy(dtype=float)
        )
        y = np.log(errors)
        design = np.column_stack([np.ones(len(x), dtype=float), x])
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        _require(int(rank) == 2, "P6 regression is rank deficient")
        fitted = design @ coefficients
        total = float(np.sum((y - np.mean(y)) ** 2))
        residual = float(np.sum((y - fitted) ** 2))
        _require(total > 0.0 and np.isfinite(residual), "invalid P6 regression variance")
        r_squared = float(1.0 - residual / total)
        slope = float(coefficients[1])
        output[str(teacher_n)][root] = {
            "slope": slope,
            "intercept": float(coefficients[0]),
            "r_squared": r_squared,
            "n_cells": int(len(rows)),
            "passed": bool(slope < 0.0 and r_squared >= minimum_r_squared),
        }
    return output


def _gate_sensitivity(
    suffix: pd.DataFrame,
    gate: Mapping[str, Any],
    *,
    mse_factor: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = gate["frozen_decision_config"]
    tolerance = float(config["tolerance_nrmse"])
    min_variance = float(config["min_target_variance"])
    minimum_successes = int(config["min_root_successes"])
    history = suffix[suffix["model"].astype(str).eq("rr_same_identity")].copy()
    history["registered_numerator_nrmse"] = np.sqrt(
        history["history_mse"] / history["target_variance"]
    )
    history["calibrated_history_nrmse"] = np.sqrt(
        mse_factor * history["history_mse"] / history["target_variance"]
    )
    rank_vectors_identical = True
    for _, cell in history.groupby(["root_id", "system_id"], sort=True):
        recorded_rank = cell["history_nrmse"].rank(method="average").to_numpy()
        calibrated_rank = cell["calibrated_history_nrmse"].rank(
            method="average"
        ).to_numpy()
        if not np.array_equal(recorded_rank, calibrated_rank):
            rank_vectors_identical = False
            break
    _require(rank_vectors_identical, "G2 history rank vector changed numerically")

    g3_families: dict[str, Any] = {}
    for teacher_n in TEACHER_NS:
        root_success: dict[str, bool] = {}
        for root in ROOTS:
            rows = history[
                history["teacher_n_qubits"].eq(int(teacher_n))
                & history["root_id"].astype(str).eq(root)
            ]
            live = rows["target_variance"].to_numpy(dtype=float) >= min_variance
            calibrated = rows["calibrated_history_nrmse"].to_numpy(dtype=float)
            root_success[root] = bool(
                np.any(live & (calibrated <= tolerance))
                and np.any(live & (calibrated > tolerance))
            )
        successes = int(sum(root_success.values()))
        g3_families[str(teacher_n)] = {
            "successful_roots": successes,
            "total_roots": len(ROOTS),
            "minimum_successful_roots": minimum_successes,
            "passed": bool(successes >= minimum_successes),
        }
    g3_passed = bool(all(value["passed"] for value in g3_families.values()))

    g4_gammas = tuple(float(value) for value in config["g4_gammas"])
    transition_cells = 0
    root_transition: dict[str, bool] = {}
    long_values: list[float] = []
    short_values: list[float] = []
    for root in ROOTS:
        root_checks: list[bool] = []
        for teacher_n, gamma in product(TEACHER_NS, g4_gammas):
            cell = history[
                history["root_id"].astype(str).eq(root)
                & history["teacher_n_qubits"].eq(int(teacher_n))
                & history["teacher_gamma"].eq(float(gamma))
                & history["window"].isin([min(WINDOWS), max(WINDOWS)])
            ]
            _require(len(cell) == 2, "G4 transition cell is incomplete")
            small = cell.loc[cell["window"].eq(min(WINDOWS))].iloc[0]
            large = cell.loc[cell["window"].eq(max(WINDOWS))].iloc[0]
            short = float(small["calibrated_history_nrmse"])
            long = float(large["calibrated_history_nrmse"])
            passed = bool(
                float(small["target_variance"]) >= min_variance
                and float(large["target_variance"]) >= min_variance
                and short > tolerance
                and long <= tolerance
            )
            transition_cells += int(passed)
            root_checks.append(passed)
            short_values.append(short)
            long_values.append(long)
        root_transition[root] = bool(all(root_checks))
    g4_roots = int(sum(root_transition.values()))
    g4_report = gate["primary_gates"]["G4"]

    g5_gamma = float(config["g5_gamma"])
    fixed_history = history[
        history["teacher_gamma"].eq(g5_gamma)
        & history["window"].eq(max(WINDOWS))
    ]
    _require(len(fixed_history) == len(ROOTS) * len(TEACHER_NS), "G5 history grid differs")
    feasible = (
        fixed_history["target_variance"].ge(min_variance)
        & fixed_history["calibrated_history_nrmse"].le(tolerance)
    )
    feasible_cells = int(feasible.sum())

    ptd = suffix[suffix["model"].astype(str).eq("ptd_parallel")].copy()
    registered_root_gains, registered_root_n_gains = _g5_gains(
        ptd,
        metric="representation_nrmse",
        gamma=g5_gamma,
        window=max(WINDOWS),
    )
    registered_values = np.asarray(list(registered_root_gains.values()), dtype=float)
    g5_report = gate["primary_gates"]["G5"]
    for root, value in registered_root_gains.items():
        _require(
            _close(value, g5_report["root_gains"][root]),
            f"registered G5 root gain differs for {root}",
        )
    _require(
        _close(float(np.median(registered_values)), g5_report["median_gain"]),
        "registered G5 median cannot be reproduced",
    )

    suffix_residual = (
        suffix["total_mse"].to_numpy(dtype=float)
        - mse_factor * suffix["history_mse"].to_numpy(dtype=float)
    )
    negative_all = int(np.sum(suffix_residual < 0.0))
    ptd["counterfactual_residual_mse"] = (
        ptd["total_mse"] - mse_factor * ptd["history_mse"]
    )
    fixed_mask = (
        ptd["teacher_gamma"].eq(g5_gamma)
        & ptd["window"].eq(max(WINDOWS))
        & ptd["snapshot_budget"].isin(
            [min(SNAPSHOT_BUDGETS), max(SNAPSHOT_BUDGETS)]
        )
    )
    fixed_g5 = ptd.loc[fixed_mask]
    negative_fixed = int(np.sum(fixed_g5["counterfactual_residual_mse"] < 0.0))
    _require(negative_fixed == 0, "counterfactual G5 residual contains negatives")
    ptd["counterfactual_residual_nrmse"] = np.where(
        ptd["counterfactual_residual_mse"] >= 0.0,
        np.sqrt(
            np.maximum(ptd["counterfactual_residual_mse"], 0.0)
            / ptd["target_variance"]
        ),
        np.nan,
    )
    corrected_root_gains, corrected_root_n_gains = _g5_gains(
        ptd,
        metric="counterfactual_residual_nrmse",
        gamma=g5_gamma,
        window=max(WINDOWS),
    )
    corrected_values = np.asarray(list(corrected_root_gains.values()), dtype=float)

    family_registered: dict[str, Any] = {}
    family_corrected: dict[str, Any] = {}
    for teacher_n in TEACHER_NS:
        registered = np.asarray(
            [registered_root_n_gains[root][str(teacher_n)] for root in ROOTS],
            dtype=float,
        )
        corrected = np.asarray(
            [corrected_root_n_gains[root][str(teacher_n)] for root in ROOTS],
            dtype=float,
        )
        gate_family = gate["primary_gates"]["G6"]["families"][str(teacher_n)]["g5"]
        _require(
            _close(float(np.median(registered)), gate_family["median_gain"]),
            f"registered G6 family {teacher_n} median cannot be reproduced",
        )
        family_registered[str(teacher_n)] = {
            "median_root_gain": float(np.median(registered)),
            "positive_roots": int(np.sum(registered > 0.0)),
            "total_roots": int(len(registered)),
            "passed": bool(
                np.median(registered) >= float(config["min_feature_gain"])
                and np.sum(registered > 0.0) >= minimum_successes
            ),
        }
        family_corrected[str(teacher_n)] = {
            "median_root_gain": float(np.median(corrected)),
            "positive_roots": int(np.sum(corrected > 0.0)),
            "total_roots": int(len(corrected)),
            "passed": bool(
                np.median(corrected) >= float(config["min_feature_gain"])
                and np.sum(corrected > 0.0) >= minimum_successes
            ),
        }

    p6 = gate["secondary_p6_forgetting_scaling"]
    registered_fits = _p6_fits(
        history,
        metric="registered_numerator_nrmse",
        teacher_dt=float(p6["teacher_dt"]),
        minimum_r_squared=float(p6["minimum_r_squared"]),
    )
    calibrated_fits = _p6_fits(
        history,
        metric="calibrated_history_nrmse",
        teacher_dt=float(p6["teacher_dt"]),
        minimum_r_squared=float(p6["minimum_r_squared"]),
    )
    slope_changes: list[float] = []
    r2_changes: list[float] = []
    intercept_shift_errors: list[float] = []
    report_slope_errors: list[float] = []
    report_r2_errors: list[float] = []
    expected_shift = float(math.log(math.sqrt(mse_factor)))
    for teacher_n, root in product(TEACHER_NS, ROOTS):
        registered = registered_fits[str(teacher_n)][root]
        calibrated = calibrated_fits[str(teacher_n)][root]
        reported = p6["families"][str(teacher_n)]["root_fits"][root]
        slope_changes.append(float(calibrated["slope"]) - float(registered["slope"]))
        r2_changes.append(
            float(calibrated["r_squared"]) - float(registered["r_squared"])
        )
        intercept_shift_errors.append(
            float(calibrated["intercept"])
            - float(registered["intercept"])
            - expected_shift
        )
        report_slope_errors.append(float(registered["slope"]) - float(reported["slope"]))
        report_r2_errors.append(
            float(registered["r_squared"]) - float(reported["r_squared"])
        )
    _require(_maximum_absolute(report_slope_errors) <= 2e-12, "P6 slopes differ from report")
    _require(_maximum_absolute(report_r2_errors) <= 2e-12, "P6 R-squared differs from report")
    _require(_maximum_absolute(slope_changes) <= 2e-12, "P6 slope changed under scaling")
    _require(_maximum_absolute(r2_changes) <= 2e-12, "P6 R-squared changed under scaling")
    _require(
        _maximum_absolute(intercept_shift_errors) <= 2e-12,
        "P6 intercept shift differs from log scaling factor",
    )

    g5_registered_passed = bool(
        feasible_cells == len(ROOTS) * len(TEACHER_NS)
        and np.median(registered_values) >= float(config["min_feature_gain"])
        and np.sum(registered_values > 0.0) >= minimum_successes
    )
    g5_corrected_passed = bool(
        feasible_cells == len(ROOTS) * len(TEACHER_NS)
        and np.median(corrected_values) >= float(config["min_feature_gain"])
        and np.sum(corrected_values > 0.0) >= minimum_successes
    )
    g6_registered_passed = bool(all(value["passed"] for value in family_registered.values()))
    g6_corrected_passed = bool(all(value["passed"] for value in family_corrected.values()))
    calibrated_gate_status = {
        "G0": True,
        "G1": True,
        "G2": True,
        "G3": g3_passed,
        "G4": bool(
            transition_cells == len(ROOTS) * len(TEACHER_NS) * len(g4_gammas)
            and g4_roots >= minimum_successes
            and g4_report["passed"] is True
        ),
        "G5": g5_registered_passed,
        "G6": g6_registered_passed,
    }
    _require(all(calibrated_gate_status.values()), "a primary gate fails after calibration")

    gate_summary = {
        "registered_decision": str(gate["decision"]),
        "calibrated_sensitivity_decision": "CONFIRMS_FINITE_RESOURCE_FRONTIER",
        "decision_invariant": True,
        "primary_gate_status_after_history_calibration": calibrated_gate_status,
        "G0_G1": {
            "status": "REGISTERED_EMPIRICAL_AUDIT_RETAINED",
            "reason": "The calibrated numerator does not overwrite the exact realized-panel decomposition.",
        },
        "G2": {
            "status": "EXACTLY_INVARIANT_UNDER_POSITIVE_COMMON_SCALING",
            "reason": "Multiplication by one positive constant preserves all ranks and ties.",
            "history_rank_vectors_identical_after_numeric_reconstruction": (
                rank_vectors_identical
            ),
            "median_root_median_spearman": float(
                gate["primary_gates"]["G2"]["median_root_median_spearman"]
            ),
            "successful_roots": int(gate["primary_gates"]["G2"]["successful_roots"]),
            "total_roots": len(ROOTS),
            "passed": True,
        },
        "G3": {
            "families": g3_families,
            "passed": g3_passed,
        },
        "G4": {
            "calibrated_transition_cells": transition_cells,
            "total_transition_cells": len(ROOTS) * len(TEACHER_NS) * len(g4_gammas),
            "calibrated_transition_roots": g4_roots,
            "total_roots": len(ROOTS),
            "minimum_calibrated_short_window_history_nrmse": float(min(short_values)),
            "maximum_calibrated_long_window_history_nrmse": float(max(long_values)),
            "stream_gain_unchanged": True,
            "median_stream_gain": float(g4_report["median_gain"]),
            "positive_stream_gain_roots": int(g4_report["positive_roots"]),
            "passed": calibrated_gate_status["G4"],
        },
        "G5": {
            "calibrated_history_feasible_root_family_cells": feasible_cells,
            "total_root_family_cells": len(ROOTS) * len(TEACHER_NS),
            "maximum_calibrated_history_nrmse": float(
                fixed_history["calibrated_history_nrmse"].max()
            ),
            "registered_empirical_representation_gain": {
                "median_root_gain": float(np.median(registered_values)),
                "positive_roots": int(np.sum(registered_values > 0.0)),
                "total_roots": int(len(registered_values)),
                "unchanged_by_history_only_calibration": True,
                "passed": g5_registered_passed,
            },
            "counterfactual_total_minus_calibrated_history_gain": {
                "median_root_gain": float(np.median(corrected_values)),
                "positive_roots": int(np.sum(corrected_values > 0.0)),
                "total_roots": int(len(corrected_values)),
                "passed": g5_corrected_passed,
            },
            "passed": calibrated_gate_status["G5"],
        },
        "G6": {
            "registered_empirical_representation_gain_by_family": family_registered,
            "counterfactual_total_minus_calibrated_history_gain_by_family": family_corrected,
            "registered_passed": g6_registered_passed,
            "counterfactual_passed": g6_corrected_passed,
            "passed": calibrated_gate_status["G6"],
        },
        "P6_nondecision_bearing": {
            "status": "EXACTLY_INVARIANT_UP_TO_LOG_INTERCEPT_SHIFT",
            "fits": len(ROOTS) * len(TEACHER_NS),
            "expected_log_intercept_shift": expected_shift,
            "maximum_absolute_slope_change": _maximum_absolute(slope_changes),
            "maximum_absolute_r_squared_change": _maximum_absolute(r2_changes),
            "maximum_absolute_intercept_shift_error": _maximum_absolute(
                intercept_shift_errors
            ),
            "maximum_absolute_reported_slope_reproduction_error": _maximum_absolute(
                report_slope_errors
            ),
            "maximum_absolute_reported_r_squared_reproduction_error": _maximum_absolute(
                report_r2_errors
            ),
            "all_fits_pass": bool(
                all(
                    calibrated_fits[str(teacher_n)][root]["passed"]
                    for teacher_n, root in product(TEACHER_NS, ROOTS)
                )
            ),
        },
    }
    decomposition_boundary = {
        "registered_exact_identity_retained": True,
        "numerator_only_calibration_is_an_exact_empirical_decomposition": False,
        "counterfactual_residual_definition": (
            "registered total_mse - calibrated history_mse; diagnostic only"
        ),
        "negative_counterfactual_residual_rows_all_models": negative_all,
        "total_rows_all_models": int(len(suffix)),
        "negative_counterfactual_residual_rows_in_fixed_G5_contrast": negative_fixed,
        "fixed_G5_contrast_rows": int(len(fixed_g5)),
    }
    return gate_summary, decomposition_boundary


def analyze(
    suffix_path: Path = DEFAULT_SUFFIX,
    gate_report_path: Path = DEFAULT_GATE_REPORT,
    phase_source_path: Path = DEFAULT_PHASE_SOURCE,
) -> dict[str, Any]:
    """Validate the frozen inputs and return the deterministic companion summary."""

    paths = {
        "suffix_decomposition.csv": Path(suffix_path).resolve(),
        "canonical_gate_report.json": Path(gate_report_path).resolve(),
        "fig04_regime_map_source.csv": Path(phase_source_path).resolve(),
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in EXPECTED_SHA256.items():
        _require(
            hashes[name] == expected,
            f"{name} frozen SHA-256 mismatch: {hashes[name]} != {expected}",
        )

    gate = load_strict_json(paths["canonical_gate_report.json"])
    _validate_gate_report(gate, hashes["suffix_decomposition.csv"])
    suffix = pd.read_csv(paths["suffix_decomposition.csv"])
    config = gate["frozen_decision_config"]
    suffix_validation = validate_suffix_frame(
        suffix,
        tolerance=float(config["tolerance_nrmse"]),
        min_target_variance=float(config["min_target_variance"]),
    )
    ptd = suffix[suffix["model"].astype(str).eq("ptd_parallel")].copy()
    registered_phase = _phase_table(ptd, "regime")
    phase = pd.read_csv(paths["fig04_regime_map_source.csv"])
    phase_validation = validate_phase_source(
        phase,
        registered_phase,
        suffix_sha256=hashes["suffix_decomposition.csv"],
        gate_sha256=hashes["canonical_gate_report.json"],
        decision=str(gate["decision"]),
    )

    branches_per_group = int(suffix_validation["branches_per_group"])
    mse_factor = float(branches_per_group / (branches_per_group - 1))
    nrmse_factor = float(math.sqrt(mse_factor))
    tolerance = float(config["tolerance_nrmse"])
    ptd["calibrated_history_nrmse"] = np.sqrt(
        mse_factor * ptd["history_mse"] / ptd["target_variance"]
    )
    ptd["calibrated_regime"] = [
        classify_regime(
            history_nrmse=float(row.calibrated_history_nrmse),
            total_nrmse=float(row.total_nrmse),
            oracle_nrmse=float(row.oracle_nrmse),
            target_variance=float(row.target_variance),
            tolerance=tolerance,
            min_target_variance=float(config["min_target_variance"]),
        )
        for row in ptd.itertuples(index=False)
    ]

    all_calibrated_history = np.sqrt(
        mse_factor * suffix["history_mse"] / suffix["target_variance"]
    )
    all_calibrated_regimes = [
        classify_regime(
            history_nrmse=float(history_nrmse),
            total_nrmse=float(row.total_nrmse),
            oracle_nrmse=float(row.oracle_nrmse),
            target_variance=float(row.target_variance),
            tolerance=tolerance,
            min_target_variance=float(config["min_target_variance"]),
        )
        for history_nrmse, row in zip(
            all_calibrated_history,
            suffix.itertuples(index=False),
            strict=True,
        )
    ]
    all_changed_mask = (
        suffix["regime"].astype(str).to_numpy()
        != np.asarray(all_calibrated_regimes, dtype=object)
    )
    all_registered_changed = suffix.loc[all_changed_mask, "regime"].astype(str)
    all_calibrated_changed = pd.Series(
        np.asarray(all_calibrated_regimes, dtype=object)[all_changed_mask],
        index=all_registered_changed.index,
        dtype="object",
    )
    all_transitions = all_registered_changed + "->" + all_calibrated_changed

    root_changed = ptd[ptd["regime"].astype(str) != ptd["calibrated_regime"].astype(str)]
    transitions = (
        root_changed["regime"].astype(str)
        + "->"
        + root_changed["calibrated_regime"].astype(str)
    )
    transition_counts = {
        str(name): int(value) for name, value in transitions.value_counts().sort_index().items()
    }
    changed_from_emulated_or_representation = int(
        root_changed["regime"].isin(["EMULATED", "REPRESENTATION_LIMITED"]).sum()
    )

    calibrated_phase = _phase_table(ptd, "calibrated_regime")
    phase_compare = registered_phase.merge(
        calibrated_phase,
        on=list(PHASE_KEY),
        suffixes=("_registered", "_calibrated"),
        validate="one_to_one",
    )
    composition_changed = phase_compare[
        phase_compare["regimes_present_registered"].astype(str)
        != phase_compare["regimes_present_calibrated"].astype(str)
    ].copy()
    modal_changed = phase_compare[
        phase_compare["modal_regime_registered"].astype(str)
        != phase_compare["modal_regime_calibrated"].astype(str)
    ].copy()
    changed_cells: list[dict[str, Any]] = []
    modal_keys = {
        tuple(row[column] for column in PHASE_KEY)
        for _, row in modal_changed.iterrows()
    }
    for _, row in composition_changed.iterrows():
        key = tuple(row[column] for column in PHASE_KEY)
        changed_cells.append(
            {
                "teacher_n_qubits": int(row["teacher_n_qubits"]),
                "teacher_gamma": float(row["teacher_gamma"]),
                "window": int(row["window"]),
                "feature_budget": int(row["feature_budget"]),
                "registered_modal_regime": str(row["modal_regime_registered"]),
                "calibrated_modal_regime": str(row["modal_regime_calibrated"]),
                "registered_agreement_n": int(row["agreement_n_registered"]),
                "calibrated_agreement_n": int(row["agreement_n_calibrated"]),
                "registered_regimes_present": str(row["regimes_present_registered"]),
                "calibrated_regimes_present": str(row["regimes_present_calibrated"]),
                "modal_label_changed": key in modal_keys,
            }
        )

    gate_sensitivity, decomposition_boundary = _gate_sensitivity(
        suffix,
        gate,
        mse_factor=mse_factor,
    )
    source_artifacts = {
        name: {
            "sha256": hashes[name],
            **(
                {"rows": int(len(suffix))}
                if name == "suffix_decomposition.csv"
                else {"rows": int(len(phase))}
                if name == "fig04_regime_map_source.csv"
                else {}
            ),
        }
        for name in (
            "suffix_decomposition.csv",
            "canonical_gate_report.json",
            "fig04_regime_map_source.csv",
        )
    }
    return {
        "schema_version": 1,
        "study": "finite_k_history_floor_calibration_companion",
        "status": "POST_HOC_NON_DECISION_BEARING_SENSITIVITY",
        "claim_boundary": {
            "registered_estimand": "exact realized-panel suffix risk with divisor K",
            "calibrated_estimand": (
                "population-prefix conditional-variance numerator under conditionally iid prefix branches"
            ),
            "conditional_iid_assumption_required": True,
            "history_nrmse_ratio_claimed_unbiased": False,
            "finite_suffix_group_sampling_G_corrected": False,
            "registered_target_variance_held_fixed": True,
            "registered_empirical_decomposition_overwritten": False,
            "registered_gate_report_overwritten": False,
        },
        "calibration": {
            "branches_per_suffix_group": branches_per_group,
            "expected_recorded_fraction_of_population_history_mse": float(
                (branches_per_group - 1) / branches_per_group
            ),
            "finite_K_downward_bias_fraction": float(1.0 / branches_per_group),
            "history_mse_numerator_factor": mse_factor,
            "history_nrmse_factor_holding_registered_target_variance": nrmse_factor,
            "registered_history_tolerance_nrmse": tolerance,
            "equivalent_registered_scale_tolerance_nrmse": float(tolerance / nrmse_factor),
        },
        "source_artifacts": source_artifacts,
        "input_validation": {
            "suffix_decomposition": suffix_validation,
            "phase_source": phase_validation,
            "strict_gate_json": True,
            "all_frozen_sha256_match": True,
        },
        "root_level_phase_labels": {
            "scope": "PTD root-resource rows used to construct the 144-cell phase map",
            "rows": int(len(ptd)),
            "registered_counts": _regime_counts(ptd["regime"].astype(str)),
            "calibrated_counts": _regime_counts(ptd["calibrated_regime"].astype(str)),
            "changed_rows": int(len(root_changed)),
            "transition_counts": transition_counts,
            "rows_changed_from_registered_emulated_or_representation_limited": (
                changed_from_emulated_or_representation
            ),
        },
        "all_model_row_label_sensitivity": {
            "scope": (
                "All 5,760 model rows; repeated history panels make these diagnostic rows, "
                "not independent statistical units or phase-map cells."
            ),
            "rows": int(len(suffix)),
            "changed_rows": int(np.sum(all_changed_mask)),
            "transition_counts": {
                str(name): int(value)
                for name, value in all_transitions.value_counts().sort_index().items()
            },
        },
        "modal_phase_map": {
            "cells": int(len(registered_phase)),
            "registered_counts": _regime_counts(
                registered_phase["modal_regime"].astype(str), modal=True
            ),
            "calibrated_counts": _regime_counts(
                calibrated_phase["modal_regime"].astype(str), modal=True
            ),
            "composition_or_agreement_changed_cells": int(len(composition_changed)),
            "modal_label_changed_cells": int(len(modal_changed)),
            "changed_cells": changed_cells,
        },
        "gate_sensitivity": gate_sensitivity,
        "decomposition_boundary": {
            **decomposition_boundary,
            "maximum_registered_empirical_identity_error": float(
                suffix_validation["max_empirical_decomposition_error"]
            ),
        },
    }


def write_summary(summary: Mapping[str, Any], output_dir: Path) -> Path:
    """Write one strict deterministic JSON file, refusing to overwrite it."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / OUTPUT_NAME
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    payload = json.dumps(
        summary,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary = output / f".{OUTPUT_NAME}.tmp"
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite existing temporary file: {temporary}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the finite-K suffix-history numerator calibration."
    )
    parser.add_argument("--suffix", type=Path, default=DEFAULT_SUFFIX)
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--phase-source", type=Path, default=DEFAULT_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.suffix, args.gate_report, args.phase_source)
    output = write_summary(summary, args.output_dir)
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": sha256_file(output),
                "decision_invariant": summary["gate_sensitivity"]["decision_invariant"],
                "root_label_changes": summary["root_level_phase_labels"]["changed_rows"],
                "modal_label_changes": summary["modal_phase_map"]["modal_label_changed_cells"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
