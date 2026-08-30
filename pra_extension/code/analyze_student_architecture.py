#!/usr/bin/env python3
"""Strict root-level analysis for the QELM architecture-family stress test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from student_architecture import (
    ARCHITECTURES,
    default_architecture_config,
    graph_edges,
    matched_architecture_couplings,
    validate_architecture_config,
)


REQUIRED_COLUMNS = {
    "root_id",
    "teacher_n_qubits",
    "teacher_gamma",
    "student_model",
    "student_architecture",
    "student_n_qubits",
    "window",
    "snapshot_budget",
    "feature_budget",
    "graph_edge_count",
    "coupling_rescale_factor",
    "coupling_edge_l2",
    "dense_reference_edge_l2",
    "student_coupling_seed",
    "teacher_coupling_seed",
    "input_seed",
    "selected_alpha",
    "intercept_only_selected",
    "validation_nrmse",
    "test_nrmse",
    "test_mse",
    "test_target_variance",
    "feature_rank_train_validation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_median_ci(values: np.ndarray, seed: int) -> list[float]:
    array = np.asarray(values, float)
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(array, size=(50_000, len(array)), replace=True)
    estimates = np.median(draws, axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def validate_architecture_results(results: pd.DataFrame, profile: str) -> None:
    config = default_architecture_config(profile)
    validate_architecture_config(config)
    missing = REQUIRED_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    key_columns = [
        "root_id",
        "teacher_n_qubits",
        "snapshot_budget",
        "student_architecture",
    ]
    if results.duplicated(key_columns).any():
        raise ValueError("duplicate architecture result keys")

    expected = {
        (root.root_id, int(teacher_n), int(budget), architecture)
        for root in config.roots
        for teacher_n in config.teacher_n_qubits
        for budget in config.snapshot_budgets
        for architecture in (*config.architectures, "quadratic_delay")
    }
    observed = {
        (str(row.root_id), int(row.teacher_n_qubits), int(row.snapshot_budget), str(row.student_architecture))
        for row in results.itertuples(index=False)
    }
    if observed != expected:
        missing_keys = sorted(expected - observed)[:3]
        extra_keys = sorted(observed - expected)[:3]
        raise ValueError(f"Cartesian profile mismatch; missing={missing_keys}, extra={extra_keys}")

    numeric = [
        "teacher_gamma",
        "student_n_qubits",
        "window",
        "snapshot_budget",
        "feature_budget",
        "graph_edge_count",
        "coupling_rescale_factor",
        "coupling_edge_l2",
        "dense_reference_edge_l2",
        "validation_nrmse",
        "test_nrmse",
        "test_mse",
        "test_target_variance",
        "feature_rank_train_validation",
    ]
    values = results[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("numeric architecture fields must all be finite")
    if (results[["validation_nrmse", "test_nrmse", "test_mse"]].to_numpy(float) < 0).any():
        raise ValueError("error metrics must be nonnegative")
    if (results.test_target_variance.to_numpy(float) <= 0).any():
        raise ValueError("test target variance must be positive")
    reconstructed = np.sqrt(results.test_mse / results.test_target_variance)
    if not np.allclose(reconstructed, results.test_nrmse, atol=2e-12, rtol=2e-12):
        raise ValueError("NRMSE/MSE/target-variance identity failed")
    if (results.feature_rank_train_validation > results.feature_budget).any():
        raise ValueError("feature rank exceeds nominal feature budget")

    parsed_alpha = pd.to_numeric(results.selected_alpha, errors="coerce").to_numpy(float)
    if np.isnan(parsed_alpha).any() or any(
        not any(
            (np.isposinf(value) and np.isposinf(candidate))
            or np.isclose(value, candidate, atol=0.0, rtol=1e-14)
            for candidate in config.alpha_grid
        )
        for value in parsed_alpha
    ):
        raise ValueError("selected alpha lies outside declared grid")
    expected_intercept = np.isposinf(parsed_alpha)
    parsed_intercept = results.intercept_only_selected.map(
        lambda value: value if isinstance(value, bool) else str(value).lower() == "true"
    )
    if not np.array_equal(expected_intercept, parsed_intercept.to_numpy()):
        raise ValueError("intercept-only indicator disagrees with selected alpha")

    root_lookup = {root.root_id: root for root in config.roots}
    edge_counts = {
        name: len(graph_edges(name, config.student_n_qubits)) for name in config.architectures
    }
    n_observables = 3 * config.student_n_qubits
    for root_id, root_rows in results.groupby("root_id", sort=True):
        root = root_lookup[str(root_id)]
        _, _, expected_dense_norm = matched_architecture_couplings(
            root=root,
            architecture="dense",
            n_qubits=config.student_n_qubits,
            coupling_scale=config.coupling_scale,
        )
        if not np.allclose(root_rows.dense_reference_edge_l2, expected_dense_norm, atol=2e-12, rtol=2e-12):
            raise ValueError(f"dense reference norm mismatch for {root_id}")
        target_keys = ["teacher_n_qubits", "snapshot_budget"]
        for _, same_target in root_rows.groupby(target_keys, sort=True):
            if np.ptp(same_target.test_target_variance.to_numpy(float)) > 2e-15:
                raise ValueError("test target variance changed across student models")
        for row in root_rows.itertuples(index=False):
            architecture = str(row.student_architecture)
            expected_width = int(row.snapshot_budget) * n_observables
            if int(row.feature_budget) != expected_width:
                raise ValueError("feature budget is not matched across models")
            if int(row.teacher_coupling_seed) != int(root.teacher_coupling_seeds[int(row.teacher_n_qubits)]):
                raise ValueError("teacher seed mismatch")
            if int(row.input_seed) != int(root.input_seed):
                raise ValueError("input seed mismatch")
            if architecture == "quadratic_delay":
                if str(row.student_model) != "quadratic_delay":
                    raise ValueError("quadratic control model label mismatch")
                if any(
                    float(value) != 0.0
                    for value in (
                        row.student_n_qubits,
                        row.graph_edge_count,
                        row.coupling_rescale_factor,
                        row.coupling_edge_l2,
                        row.student_coupling_seed,
                    )
                ):
                    raise ValueError("quadratic control carries quantum architecture fields")
                continue
            if str(row.student_model) != "qelm":
                raise ValueError("QELM architecture has wrong model label")
            if int(row.student_n_qubits) != config.student_n_qubits:
                raise ValueError("student register size mismatch")
            if int(row.graph_edge_count) != edge_counts[architecture]:
                raise ValueError("graph edge count mismatch")
            if int(row.student_coupling_seed) != int(root.student_coupling_seeds[architecture]):
                raise ValueError("student coupling seed mismatch")
            _, expected_scale, expected_norm = matched_architecture_couplings(
                root=root,
                architecture=architecture,
                n_qubits=config.student_n_qubits,
                coupling_scale=config.coupling_scale,
                dense_reference_norm=expected_dense_norm,
            )
            if not np.isclose(float(row.coupling_rescale_factor), expected_scale, atol=2e-12, rtol=2e-12):
                raise ValueError("coupling rescale factor mismatch")
            if not np.isclose(float(row.coupling_edge_l2), expected_norm, atol=2e-12, rtol=2e-12):
                raise ValueError("coupling norm mismatch")
            if not np.isclose(expected_norm, expected_dense_norm, atol=2e-12, rtol=2e-12):
                raise ValueError("coupling norm matching failed")


def analyze(results: pd.DataFrame, profile: str) -> tuple[dict[str, Any], pd.DataFrame]:
    validate_architecture_results(results, profile)
    config = default_architecture_config(profile)
    low, high = min(config.snapshot_budgets), max(config.snapshot_budgets)
    qelm_rows = results[results.student_model.eq("qelm")].copy()
    pivot = qelm_rows.pivot(
        index=["root_id", "teacher_n_qubits", "student_architecture"],
        columns="snapshot_budget",
        values="test_nrmse",
    )
    pivot["feature_gain_percent"] = 100.0 * (pivot[low] - pivot[high]) / pivot[low]
    root_arch = (
        pivot["feature_gain_percent"]
        .groupby(["root_id", "student_architecture"])
        .median()
        .rename("median_teacher_feature_gain_percent")
        .reset_index()
    )
    architecture_effects: dict[str, Any] = {}
    for index, architecture in enumerate(ARCHITECTURES):
        values = root_arch.loc[
            root_arch.student_architecture.eq(architecture),
            "median_teacher_feature_gain_percent",
        ].to_numpy(float)
        high_rows = qelm_rows[
            qelm_rows.student_architecture.eq(architecture)
            & qelm_rows.snapshot_budget.eq(high)
        ]
        architecture_effects[architecture] = {
            "median_root_feature_gain_percent": float(np.median(values)),
            "bootstrap_95_ci_percent": _bootstrap_median_ci(values, 86_100 + index),
            "positive_roots": int(np.sum(values > 0.0)),
            "n_roots": int(len(values)),
            "positive_root_teacher_cells": int(
                np.sum(
                    pivot.xs(architecture, level="student_architecture")[
                        "feature_gain_percent"
                    ].to_numpy(float)
                    > 0.0
                )
            ),
            "n_root_teacher_cells": int(len(high_rows)),
            "median_high_budget_test_nrmse": float(np.median(high_rows.test_nrmse)),
            "coupling_rescale_factor_range": [
                float(high_rows.coupling_rescale_factor.min()),
                float(high_rows.coupling_rescale_factor.max()),
            ],
        }

    high_qelm = qelm_rows[qelm_rows.snapshot_budget.eq(high)].sort_values(
        ["root_id", "teacher_n_qubits", "validation_nrmse", "student_architecture"],
        kind="mergesort",
    )
    selected = high_qelm.groupby(["root_id", "teacher_n_qubits"], sort=True).head(1).copy()
    selected = selected.rename(
        columns={
            "student_architecture": "selected_architecture",
            "validation_nrmse": "selected_validation_nrmse",
            "test_nrmse": "selected_qelm_test_nrmse",
        }
    )
    control = results[
        results.student_model.eq("quadratic_delay")
        & results.snapshot_budget.eq(high)
    ][["root_id", "teacher_n_qubits", "validation_nrmse", "test_nrmse"]].rename(
        columns={
            "validation_nrmse": "quadratic_validation_nrmse",
            "test_nrmse": "quadratic_test_nrmse",
        }
    )
    comparison = selected[
        [
            "root_id",
            "teacher_n_qubits",
            "selected_architecture",
            "selected_validation_nrmse",
            "selected_qelm_test_nrmse",
        ]
    ].merge(control, on=["root_id", "teacher_n_qubits"], validate="one_to_one")
    comparison["selected_qelm_gain_over_quadratic_percent"] = 100.0 * (
        comparison.quadratic_test_nrmse - comparison.selected_qelm_test_nrmse
    ) / comparison.quadratic_test_nrmse
    root_comparison = (
        comparison.groupby("root_id", sort=True)[
            "selected_qelm_gain_over_quadratic_percent"
        ]
        .median()
        .rename("median_teacher_selected_qelm_gain_over_quadratic_percent")
    )

    fixed_best = high_qelm.pivot(
        index=["root_id", "teacher_n_qubits"],
        columns="student_architecture",
        values="test_nrmse",
    )
    control_indexed = control.set_index(["root_id", "teacher_n_qubits"])
    control_beats_all = control_indexed.quadratic_test_nrmse < fixed_best.min(axis=1)
    selected_counts = {
        str(name): int(count)
        for name, count in comparison.selected_architecture.value_counts().sort_index().items()
    }
    root_values = root_comparison.to_numpy(float)
    summary = {
        "schema_version": 1,
        "study": "pra_qelm_student_architecture_family_analysis",
        "status": "EXPLORATORY_STRESS_TEST",
        "selection_guard": (
            "Ridge alpha and the QELM topology are selected on validation rows only; "
            "test targets are not used for either selection."
        ),
        "claim_boundary": (
            "The result probes four norm-matched graph topologies for one six-qubit "
            "unitary QELM family. It is not an architecture-universal statement or "
            "evidence of quantum advantage; the quadratic control remains stronger."
        ),
        "counts": {
            "rows": int(len(results)),
            "roots": int(results.root_id.nunique()),
            "root_teacher_cells": int(comparison.shape[0]),
        },
        "architecture_feature_effects": architecture_effects,
        "validation_selected_architecture_counts": selected_counts,
        "validation_selected_qelm_vs_quadratic": {
            "median_root_gain_percent": float(np.median(root_values)),
            "bootstrap_95_ci_percent": _bootstrap_median_ci(root_values, 86_200),
            "positive_roots": int(np.sum(root_values > 0.0)),
            "n_roots": int(len(root_values)),
            "positive_root_teacher_cells": int(
                np.sum(comparison.selected_qelm_gain_over_quadratic_percent > 0.0)
            ),
            "n_root_teacher_cells": int(len(comparison)),
            "quadratic_beats_every_fixed_qelm_topology_cells": int(np.sum(control_beats_all)),
            "total_fixed_topology_comparison_cells": int(len(control_beats_all)),
        },
        "audit": {
            "exact_cartesian_profile": True,
            "globally_unique_explicit_seeds": True,
            "dense_reproduces_canonical_random_couplings": True,
            "sparse_interaction_norms_match_paired_dense": True,
            "test_targets_unused_for_model_selection": True,
        },
    }
    return summary, comparison.sort_values(
        ["root_id", "teacher_n_qubits"], kind="mergesort"
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    analysis_path = output / "student_architecture_analysis.json"
    effects_path = output / "student_architecture_selected_effects.csv"
    hash_path = output / "artifact_sha256.json"
    if any(path.exists() for path in (analysis_path, effects_path, hash_path)):
        raise FileExistsError("refusing to overwrite an existing architecture analysis")

    results_path = args.results.resolve()
    results = pd.read_csv(results_path)
    summary, effects = analyze(results, args.profile)
    summary["source_results_sha256"] = sha256_file(results_path)
    effects.to_csv(effects_path, index=False)
    analysis_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    hashes = {
        path.name: sha256_file(path) for path in (analysis_path, effects_path)
    }
    hash_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["architecture_feature_effects"], indent=2, sort_keys=True))
    print(json.dumps(summary["validation_selected_qelm_vs_quadratic"], indent=2, sort_keys=True))
    print(json.dumps(hashes, sort_keys=True))


if __name__ == "__main__":
    main()
