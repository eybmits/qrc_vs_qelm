"""Post-confirmatory finite-shot errors-in-variables (EIV) experiment.

Unlike :mod:`finite_shots`, which fits a readout to exact QELM features and
adds the exact frozen-readout shot penalty, this module samples every QELM
feature used by the fit.  Train, validation, and test feature matrices use
disjoint joint-multinomial streams.  For every Monte Carlo replicate, ridge
strength is selected on noisy validation features, and the feature budget is
also selected on validation before the test score is exposed.

The teacher targets remain exact.  This is therefore an errors-in-variables
stress test for QELM measurements, not a full hardware experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from finite_shots import (
    AXES,
    ShotRoot,
    ShotStudyConfig,
    _teacher_and_student_data,
    equal_allocation,
    joint_setting_probabilities,
    outcome_table,
    validate_shot_config,
)
from qrc_vs_qelm import canonical_emulation, emulation


EIV_NAMESPACE_CODE = 28_082_602
EIV_SPLIT_CODES = {"train": 1, "validation": 2, "test": 3}


@dataclass(frozen=True)
class NoisyReadout:
    """A ridge readout selected using noisy train and validation features."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray
    alpha: float
    alpha_index: int
    validation_mse: float
    validation_alpha_mse: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (
            np.asarray(features, float) - self.feature_mean
        ) / self.feature_scale
        return standardized @ self.weights + self.intercept


def default_eiv_config(profile: str = "full") -> ShotStudyConfig:
    """Return the fresh four-qubit-teacher EIV profile."""

    alphas = (
        0.0,
        1e-10,
        1e-8,
        1e-6,
        1e-4,
        1e-2,
        1.0,
        1e2,
        1e4,
        1e6,
        float("inf"),
    )
    if profile == "smoke":
        config = ShotStudyConfig(
            roots=(
                ShotRoot(
                    root_id="eivn4_smoke01",
                    teacher_coupling_seeds={4: 9_640_104},
                    student_coupling_seed=9_641_101,
                    input_seed=9_642_101,
                    monte_carlo_seed=9_643_101,
                ),
            ),
            teacher_n_qubits=(4,),
            teacher_gamma=4.0,
            window=2,
            train_rows=24,
            validation_rows=12,
            test_rows=16,
            burnin=8,
            quantize_levels=9,
            snapshot_budgets=(1, 2, 4),
            total_shots=(300, 1_200),
            alpha_grid=alphas,
            monte_carlo_repeats=2,
        )
    elif profile == "full":
        roots = tuple(
            ShotRoot(
                root_id=f"eivn4_{index:02d}",
                teacher_coupling_seeds={4: 9_600_000 + 100 * index + 4},
                student_coupling_seed=9_610_000 + 100 * index + 1,
                input_seed=9_620_000 + 100 * index + 1,
                monte_carlo_seed=9_630_000 + 100 * index + 1,
            )
            for index in range(1, 13)
        )
        config = ShotStudyConfig(
            roots=roots,
            teacher_n_qubits=(4,),
            teacher_gamma=4.0,
            window=6,
            train_rows=128,
            validation_rows=48,
            test_rows=96,
            burnin=48,
            quantize_levels=33,
            snapshot_budgets=(1, 2, 4),
            total_shots=(300, 1_200, 4_800, 19_200),
            alpha_grid=alphas,
            monte_carlo_repeats=16,
        )
    else:
        raise ValueError("profile must be 'smoke' or 'full'")
    validate_shot_config(config)
    eiv_seed_manifest(config)
    return config


def eiv_entropy_tuple(
    *,
    root: ShotRoot,
    snapshot_budget: int,
    total_shots: int,
    replicate: int,
    split: str,
) -> tuple[int, ...]:
    """Return the auditable SeedSequence entropy for one noisy feature split."""

    if split not in EIV_SPLIT_CODES:
        raise ValueError("split must be train, validation, or test")
    values = (
        int(root.monte_carlo_seed),
        int(EIV_NAMESPACE_CODE),
        4,
        int(snapshot_budget),
        int(total_shots),
        int(replicate),
        int(EIV_SPLIT_CODES[split]),
    )
    if any(value < 0 for value in values):
        raise ValueError("EIV entropy coordinates must be nonnegative")
    return values


def eiv_seed_manifest(config: ShotStudyConfig) -> tuple[dict[str, Any], ...]:
    """Build and validate all train/validation/test sampling streams."""

    rows: list[dict[str, Any]] = []
    seen_entropy: set[tuple[int, ...]] = set()
    seen_state: set[tuple[int, ...]] = set()
    for root, budget, total_shots, replicate, split in product(
        config.roots,
        config.snapshot_budgets,
        config.total_shots,
        range(int(config.monte_carlo_repeats)),
        EIV_SPLIT_CODES,
    ):
        entropy = eiv_entropy_tuple(
            root=root,
            snapshot_budget=int(budget),
            total_shots=int(total_shots),
            replicate=int(replicate),
            split=str(split),
        )
        state = tuple(
            int(value)
            for value in np.random.SeedSequence(entropy).generate_state(
                4, dtype=np.uint32
            )
        )
        if entropy in seen_entropy or state in seen_state:
            raise ValueError("EIV sampling streams are not globally unique")
        seen_entropy.add(entropy)
        seen_state.add(state)
        rows.append(
            {
                "root_id": root.root_id,
                "snapshot_budget": int(budget),
                "total_shots": int(total_shots),
                "replicate": int(replicate),
                "split": str(split),
                "entropy": entropy,
                "state_uint32": state,
            }
        )
    return tuple(rows)


def joint_pauli_probabilities(states: np.ndarray) -> np.ndarray:
    """Precompute one global-axis multinomial distribution per row/snapshot."""

    values = np.asarray(states, complex)
    if values.ndim != 3:
        raise ValueError("states must have shape (rows, snapshots, amplitudes)")
    n_rows, snapshots, width = values.shape
    n_qubits = int(round(np.log2(width)))
    if width != 2**n_qubits:
        raise ValueError("state width is not a qubit Hilbert-space dimension")
    probabilities = np.empty(
        (n_rows, snapshots, len(AXES), width), dtype=float
    )
    for row, snapshot, axis_index in product(
        range(n_rows), range(snapshots), range(len(AXES))
    ):
        probabilities[row, snapshot, axis_index] = (
            joint_setting_probabilities(
                values[row, snapshot], AXES[axis_index], n_qubits
            )
        )
    return probabilities


def sample_joint_pauli_features(
    probabilities: np.ndarray,
    *,
    total_shots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample snapshot-major, axis-major, qubit-major single-site features.

    Each global Pauli setting uses one multinomial draw, so the six qubit
    outcomes from that setting retain their physical within-setting
    covariance.  Settings and data rows use independent draws.
    """

    values = np.asarray(probabilities, float)
    if values.ndim != 4 or values.shape[2] != len(AXES):
        raise ValueError(
            "probabilities must have shape (rows, snapshots, 3, outcomes)"
        )
    n_rows, snapshots, _, n_outcomes = values.shape
    n_qubits = int(round(np.log2(n_outcomes)))
    if n_outcomes != 2**n_qubits:
        raise ValueError("outcome width is not a qubit Hilbert-space dimension")
    if np.any(values < -1e-14) or not np.allclose(
        values.sum(axis=-1), 1.0, atol=2e-12, rtol=0.0
    ):
        raise ValueError("invalid multinomial probabilities")
    n_settings = int(snapshots) * len(AXES)
    allocation = equal_allocation(int(total_shots), n_settings)
    outcomes = outcome_table(n_qubits)
    features = np.empty(
        (n_rows, int(snapshots), len(AXES), n_qubits), dtype=float
    )
    setting = 0
    for snapshot, axis_index in product(
        range(int(snapshots)), range(len(AXES))
    ):
        shots = int(allocation[setting])
        for row in range(n_rows):
            counts = rng.multinomial(
                shots, values[row, snapshot, axis_index]
            )
            features[row, snapshot, axis_index] = (
                counts @ outcomes / float(shots)
            )
        setting += 1
    return features.reshape(n_rows, -1), allocation


def select_noisy_readout(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    alphas: Sequence[float],
) -> NoisyReadout:
    """Select alpha using noisy validation only, then refit train+validation."""

    x_train = np.asarray(train_features, float)
    y_train = np.asarray(train_targets, float)
    x_validation = np.asarray(validation_features, float)
    y_validation = np.asarray(validation_targets, float)
    if x_train.ndim != 2 or x_validation.ndim != 2:
        raise ValueError("feature matrices must be two dimensional")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("train and validation feature widths differ")
    if len(x_train) != len(y_train) or len(x_validation) != len(y_validation):
        raise ValueError("features and targets must align")
    if not alphas:
        raise ValueError("alpha grid must be nonempty")

    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    train_standardized = (x_train - mean) / scale
    validation_standardized = (x_validation - mean) / scale
    scores = np.empty(len(alphas), dtype=float)
    for index, alpha_value in enumerate(alphas):
        alpha = float(alpha_value)
        weights, intercept = emulation._fit_centered_multi(
            train_standardized, y_train, alpha
        )
        prediction = validation_standardized @ weights + intercept
        scores[index] = float(np.mean((y_validation - prediction) ** 2))
    # np.argmin implements the declared earliest-grid tie rule.
    selected_index = int(np.argmin(scores))
    selected_alpha = float(alphas[selected_index])

    combined_standardized = np.vstack(
        [train_standardized, validation_standardized]
    )
    combined_targets = np.vstack([y_train, y_validation])
    combined_rows = np.arange(len(combined_standardized), dtype=int)
    weights, intercept = emulation._fit_centered_multi(
        combined_standardized,
        combined_targets,
        selected_alpha,
    )
    return NoisyReadout(
        feature_mean=mean,
        feature_scale=scale,
        weights=np.asarray(weights, float),
        intercept=np.asarray(intercept, float),
        alpha=selected_alpha,
        alpha_index=selected_index,
        validation_mse=float(scores[selected_index]),
        validation_alpha_mse=scores,
    )


def _alpha_labels(alphas: Sequence[float]) -> list[str]:
    return [
        "+inf" if np.isposinf(float(value)) else format(float(value), ".12g")
        for value in alphas
    ]


def _select_feature_budgets(results: pd.DataFrame) -> pd.DataFrame:
    """Mark exactly one validation-selected B in each root/Q/replicate group."""

    output = results.copy()
    output["selected_budget_on_validation"] = False
    keys = ["root_id", "total_shots", "replicate"]
    ordered = output.sort_values(
        keys + ["validation_mse", "snapshot_budget"], kind="mergesort"
    )
    selected = ordered.groupby(keys, sort=True).head(1).index
    output.loc[selected, "selected_budget_on_validation"] = True
    return output.sort_values(
        keys + ["snapshot_budget"], kind="mergesort"
    ).reset_index(drop=True)


def run_eiv_study(config: ShotStudyConfig) -> pd.DataFrame:
    """Run the all-splits-noisy EIV study with validation-only selection."""

    validate_shot_config(config)
    eiv_seed_manifest(config)
    if config.teacher_n_qubits != (4,):
        raise ValueError("the EIV profile requires one four-qubit teacher")
    rows: list[dict[str, Any]] = []
    n_student = int(config.student_n_qubits)
    train_stop = int(config.train_rows)
    validation_stop = train_stop + int(config.validation_rows)
    alpha_labels = _alpha_labels(config.alpha_grid)
    alpha_grid_json = json.dumps(alpha_labels, separators=(",", ":"))

    for root in config.roots:
        targets, states, features_max = _teacher_and_student_data(config, root, 4)
        probabilities_max = joint_pauli_probabilities(states)
        train = slice(0, train_stop)
        validation = slice(train_stop, validation_stop)
        test = slice(validation_stop, len(targets))
        validation_variance = float(
            np.mean(
                (
                    targets[validation]
                    - np.mean(targets[validation], axis=0)
                )
                ** 2
            )
        )
        test_variance = float(
            np.mean((targets[test] - np.mean(targets[test], axis=0)) ** 2)
        )
        if min(validation_variance, test_variance) <= 0:
            raise RuntimeError("target variance must be positive")

        for budget in config.snapshot_budgets:
            snapshot_indices = canonical_emulation.nested_snapshot_columns(
                max_snapshots=int(config.max_snapshots),
                selected_snapshots=int(budget),
                n_observables=1,
            )
            exact_features = canonical_emulation.select_nested_snapshots(
                features_max,
                max_snapshots=int(config.max_snapshots),
                selected_snapshots=int(budget),
                n_observables=len(AXES) * n_student,
            )
            probabilities = probabilities_max[:, snapshot_indices]
            exact_reference = select_noisy_readout(
                train_features=exact_features[train],
                train_targets=targets[train],
                validation_features=exact_features[validation],
                validation_targets=targets[validation],
                alphas=config.alpha_grid,
            )
            reference_prediction = exact_reference.predict(exact_features[test])
            reference_test_mse = float(
                np.mean((targets[test] - reference_prediction) ** 2)
            )

            for total_shots, replicate in product(
                config.total_shots,
                range(int(config.monte_carlo_repeats)),
            ):
                split_features: dict[str, np.ndarray] = {}
                split_entropies: dict[str, tuple[int, ...]] = {}
                split_slices = {
                    "train": train,
                    "validation": validation,
                    "test": test,
                }
                allocation: np.ndarray | None = None
                for split_name, split_slice in split_slices.items():
                    entropy = eiv_entropy_tuple(
                        root=root,
                        snapshot_budget=int(budget),
                        total_shots=int(total_shots),
                        replicate=int(replicate),
                        split=split_name,
                    )
                    rng = np.random.default_rng(np.random.SeedSequence(entropy))
                    sampled, split_allocation = sample_joint_pauli_features(
                        probabilities[split_slice],
                        total_shots=int(total_shots),
                        rng=rng,
                    )
                    if allocation is None:
                        allocation = split_allocation
                    elif not np.array_equal(allocation, split_allocation):
                        raise RuntimeError("split allocations unexpectedly differ")
                    split_features[split_name] = sampled
                    split_entropies[split_name] = entropy
                assert allocation is not None

                model = select_noisy_readout(
                    train_features=split_features["train"],
                    train_targets=targets[train],
                    validation_features=split_features["validation"],
                    validation_targets=targets[validation],
                    alphas=config.alpha_grid,
                )
                noisy_prediction = model.predict(split_features["test"])
                exact_feature_prediction = model.predict(exact_features[test])
                test_noisy_mse = float(
                    np.mean((targets[test] - noisy_prediction) ** 2)
                )
                test_exact_feature_mse = float(
                    np.mean((targets[test] - exact_feature_prediction) ** 2)
                )
                rows.append(
                    {
                        "status": "EXPLORATORY_POST_CONFIRMATORY",
                        "root_id": root.root_id,
                        "teacher_n_qubits": 4,
                        "teacher_gamma": float(config.teacher_gamma),
                        "student_n_qubits": n_student,
                        "window": int(config.window),
                        "train_rows": int(config.train_rows),
                        "validation_rows": int(config.validation_rows),
                        "test_rows": int(config.test_rows),
                        "replicate": int(replicate),
                        "snapshot_budget": int(budget),
                        "feature_budget": int(exact_features.shape[1]),
                        "snapshot_indices_zero_based": json.dumps(
                            [int(value) for value in snapshot_indices],
                            separators=(",", ":"),
                        ),
                        "active_settings": int(3 * budget),
                        "total_shots": int(total_shots),
                        "allocation_mode": "equal_fixed_total",
                        "allocation_json": json.dumps(
                            [int(value) for value in allocation],
                            separators=(",", ":"),
                        ),
                        "minimum_setting_shots": int(np.min(allocation)),
                        "maximum_setting_shots": int(np.max(allocation)),
                        "train_entropy": json.dumps(
                            [int(value) for value in split_entropies["train"]],
                            separators=(",", ":"),
                        ),
                        "validation_entropy": json.dumps(
                            [
                                int(value)
                                for value in split_entropies["validation"]
                            ],
                            separators=(",", ":"),
                        ),
                        "test_entropy": json.dumps(
                            [int(value) for value in split_entropies["test"]],
                            separators=(",", ":"),
                        ),
                        "alpha_grid_json": alpha_grid_json,
                        "selected_alpha_index": int(model.alpha_index),
                        "selected_alpha": alpha_labels[model.alpha_index],
                        "intercept_only_selected": bool(
                            np.isposinf(model.alpha)
                        ),
                        "validation_alpha_mse_json": json.dumps(
                            [float(value) for value in model.validation_alpha_mse],
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                        "validation_target_variance": validation_variance,
                        "validation_mse": float(model.validation_mse),
                        "validation_nrmse": float(
                            np.sqrt(model.validation_mse / validation_variance)
                        ),
                        "test_target_variance": test_variance,
                        "test_noisy_feature_mse": test_noisy_mse,
                        "test_noisy_feature_nrmse": float(
                            np.sqrt(test_noisy_mse / test_variance)
                        ),
                        "test_exact_feature_mse_after_noisy_fit": (
                            test_exact_feature_mse
                        ),
                        "test_exact_feature_nrmse_after_noisy_fit": float(
                            np.sqrt(test_exact_feature_mse / test_variance)
                        ),
                        "noiseless_reference_alpha": alpha_labels[
                            exact_reference.alpha_index
                        ],
                        "noiseless_reference_validation_mse": float(
                            exact_reference.validation_mse
                        ),
                        "noiseless_reference_test_mse": reference_test_mse,
                        "noiseless_reference_test_nrmse": float(
                            np.sqrt(reference_test_mse / test_variance)
                        ),
                    }
                )

    return _select_feature_budgets(pd.DataFrame(rows))


def summarize_eiv_results(results: pd.DataFrame) -> dict[str, Any]:
    """Return descriptive metadata; inferential summaries live separately."""

    selected = results[results["selected_budget_on_validation"]]
    return {
        "schema_version": 1,
        "study": "finite_shots_errors_in_variables_n4",
        "status": "EXPLORATORY_POST_CONFIRMATORY",
        "claim_boundary": (
            "four-qubit local-loss teachers and one fixed six-qubit QELM; "
            "exact teacher labels; joint-multinomial shot noise on train, "
            "validation, and test QELM features; simulation rather than hardware"
        ),
        "measurement_model": (
            "one joint multinomial outcome per snapshot/global-Pauli setting; "
            "equal integer allocation at fixed total shots per input row"
        ),
        "selection_rule": (
            "within each replicate, alpha is selected from noisy validation "
            "MSE separately for each B; B is then selected by noisy validation "
            "MSE; test targets never enter either selection"
        ),
        "inference_unit": (
            "root; Monte Carlo replicates are reduced to a median within root before "
            "root-block uncertainty summaries"
        ),
        "counts": {
            "rows": int(len(results)),
            "roots": int(results["root_id"].nunique()),
            "replicates": int(results["replicate"].nunique()),
            "candidate_budgets_per_selection": 3,
            "validation_selected_rows": int(len(selected)),
        },
    }
