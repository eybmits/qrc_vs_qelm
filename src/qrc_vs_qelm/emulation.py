"""Utilities for observable QRC-to-window-model emulation studies.

The objects in this module operate only on already generated feature matrices.
They do not assume a simulator or inspect architecture labels.  The central
quantity is the sampled suffix diameter: the largest observed separation of
teacher outputs for two histories that share their last ``d`` inputs.  Every
deterministic ``d``-window predictor gives the same prediction to such a pair,
so its worst arm error is at least half of that separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from qrc_vs_qelm.recurrence_fingerprint import sample_pauli_expectations

RIDGE_LSTSQ_RCOND = 1e-6


def _matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _cube(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape (suffix_endpoints, pairs, outputs)"
        )
    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _indices(name: str, value: np.ndarray, n_rows: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional index array")
    if raw.dtype.kind not in "iu":
        numeric = np.asarray(raw, dtype=float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{name} must contain integer indices")
    array = np.asarray(raw, dtype=int)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.unique(array).size != array.size:
        raise ValueError(f"{name} must not contain duplicates")
    if array.min() < 0 or array.max() >= int(n_rows):
        raise ValueError(f"{name} contains indices outside the data rows")
    return array


def row_rms(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return per-row root-mean-square distances over output coordinates."""

    left = _matrix("first", first)
    right = _matrix("second", second)
    if left.shape != right.shape:
        raise ValueError("first and second must have identical shapes")
    return np.sqrt(np.mean((left - right) ** 2, axis=1))


def suffix_diameter(outputs_a: np.ndarray, outputs_b: np.ndarray) -> float:
    """Largest sampled RMS separation among common-suffix history pairs."""

    return float(np.max(row_rms(outputs_a, outputs_b)))


def window_lower_bound(outputs_a: np.ndarray, outputs_b: np.ndarray) -> float:
    """Sampled lower bound for every deterministic window-only predictor."""

    return 0.5 * suffix_diameter(outputs_a, outputs_b)


@dataclass(frozen=True)
class PairedWindowCertificate:
    """Observable common-suffix certificate for one explicit window length."""

    suffix_diameter: float
    lower_bound: float
    rms_suffix_separation: float
    paired_rmse_lower_bound: float
    worst_arm_error: float
    mean_arm_error: float
    paired_rmse: float
    prediction_disagreement: float
    bound_slack: float
    paired_rmse_slack: float
    bound_satisfied: bool


@dataclass(frozen=True)
class TailSuffixCertificate:
    """Monotone sampled witness using every registered suffix endpoint >= d."""

    suffix_diameter: float
    lower_bound: float
    rms_suffix_separation: float
    paired_rmse_lower_bound: float
    rms_witness_endpoint: int
    worst_arm_error: float
    paired_rmse_at_witness: float
    prediction_disagreement: float
    bound_satisfied: bool


def paired_window_certificate(
    outputs_a: np.ndarray,
    outputs_b: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray | None = None,
    *,
    atol: float = 1e-12,
) -> PairedWindowCertificate:
    """Evaluate a window predictor on paired histories sharing its full input.

    When ``predictions_b`` is omitted, ``predictions_a`` is used for both arms.
    Passing both arrays is useful for auditing that an implementation really is
    suffix-only: their disagreement should vanish at exact precision.
    """

    first = _matrix("outputs_a", outputs_a)
    second = _matrix("outputs_b", outputs_b)
    pred_a = _matrix("predictions_a", predictions_a)
    pred_b = pred_a if predictions_b is None else _matrix(
        "predictions_b", predictions_b
    )
    if not (first.shape == second.shape == pred_a.shape == pred_b.shape):
        raise ValueError("all paired output and prediction arrays must match")
    if not np.isfinite(atol) or float(atol) < 0.0:
        raise ValueError("atol must be finite and non-negative")

    diameter = suffix_diameter(first, second)
    lower = 0.5 * diameter
    first_errors = row_rms(first, pred_a)
    second_errors = row_rms(second, pred_b)
    arm_errors = np.concatenate([first_errors, second_errors])
    worst = float(np.max(arm_errors))
    rms_separation = float(np.sqrt(np.mean((first - second) ** 2)))
    paired_lower = 0.5 * rms_separation
    paired_rmse = float(
        np.sqrt(
            0.5
            * (
                np.mean((first - pred_a) ** 2)
                + np.mean((second - pred_b) ** 2)
            )
        )
    )
    disagreement = float(np.max(row_rms(pred_a, pred_b)))
    slack = worst - lower
    paired_slack = paired_rmse - paired_lower
    # The mathematical lower bound requires identical predictions for the two
    # histories.  A nonzero disagreement is therefore an implementation audit
    # failure, not evidence against the bound.
    satisfied = bool(
        disagreement <= float(atol)
        and slack >= -float(atol)
        and paired_slack >= -float(atol)
    )
    return PairedWindowCertificate(
        suffix_diameter=diameter,
        lower_bound=lower,
        rms_suffix_separation=rms_separation,
        paired_rmse_lower_bound=paired_lower,
        worst_arm_error=worst,
        mean_arm_error=float(np.mean(arm_errors)),
        paired_rmse=paired_rmse,
        prediction_disagreement=disagreement,
        bound_slack=float(slack),
        paired_rmse_slack=float(paired_slack),
        bound_satisfied=satisfied,
    )


def tail_suffix_certificate(
    suffix_endpoints: Sequence[int],
    outputs_a: np.ndarray,
    outputs_b: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray | None = None,
    *,
    atol: float = 1e-12,
) -> TailSuffixCertificate:
    """Certify one window using all sampled pairs at every endpoint in its tail.

    Arrays have shape ``(n_endpoints, n_pairs, n_outputs)``.  The maximum-arm
    witness pools every endpoint.  The paired-RMSE witness selects the endpoint
    with the largest RMS separation across complete history pairs.  Removing
    early endpoints as ``d`` grows makes both sampled separation witnesses
    monotone by construction.
    """

    endpoints = tuple(int(value) for value in suffix_endpoints)
    if not endpoints or endpoints != tuple(sorted(set(endpoints))):
        raise ValueError("suffix_endpoints must be non-empty, unique, and sorted")
    first = _cube("outputs_a", outputs_a)
    second = _cube("outputs_b", outputs_b)
    pred_a = _cube("predictions_a", predictions_a)
    pred_b = pred_a if predictions_b is None else _cube(
        "predictions_b", predictions_b
    )
    if len(endpoints) != first.shape[0]:
        raise ValueError("suffix_endpoints length must match the first array axis")
    if not (first.shape == second.shape == pred_a.shape == pred_b.shape):
        raise ValueError("all tail output and prediction arrays must match")
    if not np.isfinite(atol) or float(atol) < 0.0:
        raise ValueError("atol must be finite and non-negative")

    flat_shape = (-1, first.shape[-1])
    flat_first = first.reshape(flat_shape)
    flat_second = second.reshape(flat_shape)
    flat_pred_a = pred_a.reshape(flat_shape)
    flat_pred_b = pred_b.reshape(flat_shape)
    diameter = suffix_diameter(flat_first, flat_second)
    lower = 0.5 * diameter
    worst = float(
        max(
            np.max(row_rms(flat_first, flat_pred_a)),
            np.max(row_rms(flat_second, flat_pred_b)),
        )
    )
    disagreement = float(np.max(row_rms(flat_pred_a, flat_pred_b)))

    endpoint_rms = np.sqrt(np.mean((first - second) ** 2, axis=(1, 2)))
    witness_position = int(np.argmax(endpoint_rms))
    rms_separation = float(endpoint_rms[witness_position])
    paired_lower = 0.5 * rms_separation
    paired_rmse = float(
        np.sqrt(
            0.5
            * (
                np.mean(
                    (first[witness_position] - pred_a[witness_position]) ** 2
                )
                + np.mean(
                    (second[witness_position] - pred_b[witness_position]) ** 2
                )
            )
        )
    )
    satisfied = bool(
        disagreement <= float(atol)
        and worst >= lower - float(atol)
        and paired_rmse >= paired_lower - float(atol)
    )
    return TailSuffixCertificate(
        suffix_diameter=diameter,
        lower_bound=lower,
        rms_suffix_separation=rms_separation,
        paired_rmse_lower_bound=paired_lower,
        rms_witness_endpoint=endpoints[witness_position],
        worst_arm_error=worst,
        paired_rmse_at_witness=paired_rmse,
        prediction_disagreement=disagreement,
        bound_satisfied=satisfied,
    )


@dataclass(frozen=True)
class EmulationMetrics:
    """Aggregate multi-output regression errors in observable RMS units."""

    mse: float
    rmse: float
    nrmse: float
    mean_row_rms: float
    median_row_rms: float
    max_row_rms: float


def emulation_metrics(targets: np.ndarray, predictions: np.ndarray) -> EmulationMetrics:
    """Return aggregate errors for a teacher observable matrix."""

    truth = _matrix("targets", targets)
    estimate = _matrix("predictions", predictions)
    if truth.shape != estimate.shape:
        raise ValueError("targets and predictions must have identical shapes")
    squared = (truth - estimate) ** 2
    mse = float(np.mean(squared))
    centered = truth - truth.mean(axis=0, keepdims=True)
    scale = float(np.mean(centered**2))
    distances = np.sqrt(np.mean(squared, axis=1))
    return EmulationMetrics(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        nrmse=(float(np.sqrt(mse / scale)) if scale > 1e-30 else float("inf")),
        mean_row_rms=float(np.mean(distances)),
        median_row_rms=float(np.median(distances)),
        max_row_rms=float(np.max(distances)),
    )


@dataclass(frozen=True)
class MultiOutputRidge:
    """Train-standardized multi-output ridge model with an intercept."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray
    alpha: float
    validation_nrmse: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = _matrix("features", features)
        if values.shape[1] != self.feature_mean.shape[0]:
            raise ValueError("features have the wrong number of columns")
        standardized = (values - self.feature_mean) / self.feature_scale
        return standardized @ self.weights + self.intercept


def _fit_centered_multi(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = features.mean(axis=0)
    y_mean = targets.mean(axis=0)
    centered_x = features - x_mean
    centered_y = targets - y_mean
    if np.isposinf(float(alpha)):
        weights = np.zeros((features.shape[1], targets.shape[1]), dtype=float)
    else:
        # The canonical feature panels contain exact and floating-point-near
        # linear dependencies (for example repeated Pauli information across
        # temporal snapshots).  A fixed SVD cutoff prevents numerically null
        # directions from turning round-off-sized feature differences into
        # visible prediction differences.  The same resolved subspace is used
        # for every finite ridge candidate and for the registered rank audit.
        left, singular, right_t = np.linalg.svd(
            centered_x,
            full_matrices=False,
        )
        factors = np.zeros_like(singular)
        if singular.size and float(singular[0]) > 0.0:
            live = singular > float(singular[0]) * RIDGE_LSTSQ_RCOND
            if float(alpha) == 0.0:
                factors[live] = 1.0 / singular[live]
            else:
                factors[live] = singular[live] / (
                    singular[live] ** 2 + float(alpha)
                )
        weights = (right_t.T * factors) @ (left.T @ centered_y)
    intercept = y_mean - x_mean @ weights
    return weights, intercept


def fit_multivariate_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    alphas: Sequence[float],
) -> MultiOutputRidge:
    """Tune ridge on validation rows and refit on train+validation rows.

    Feature centering and scaling are fit on the training rows only and remain
    frozen for validation and the final train+validation refit.
    """

    x = _matrix("features", features)
    y = _matrix("targets", targets)
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must have the same number of rows")
    train = _indices("train_idx", train_idx, x.shape[0])
    val = _indices("val_idx", val_idx, x.shape[0])
    if np.intersect1d(train, val).size:
        raise ValueError("train_idx and val_idx must be disjoint")
    alpha_grid = tuple(float(alpha) for alpha in alphas)
    if not alpha_grid:
        raise ValueError("alphas must be non-empty")
    if any(np.isnan(alpha) or alpha < 0.0 for alpha in alpha_grid):
        raise ValueError("alphas must be non-negative numbers or positive infinity")

    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (x - mean) / scale

    best_alpha = alpha_grid[0]
    best_score = float("inf")
    for alpha in alpha_grid:
        weights, intercept = _fit_centered_multi(
            standardized[train], y[train], alpha
        )
        prediction = standardized[val] @ weights + intercept
        score = emulation_metrics(y[val], prediction).nrmse
        if score < best_score:
            best_score = score
            best_alpha = alpha

    train_val = np.concatenate([train, val])
    weights, intercept = _fit_centered_multi(
        standardized[train_val], y[train_val], best_alpha
    )
    return MultiOutputRidge(
        feature_mean=mean.copy(),
        feature_scale=scale.copy(),
        weights=np.asarray(weights, dtype=np.float64),
        intercept=np.asarray(intercept, dtype=np.float64),
        alpha=float(best_alpha),
        validation_nrmse=float(best_score),
    )


def sample_feature_matrix(
    expectations: np.ndarray,
    n_shots: int | None,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return exact or independently sampled Pauli expectation features."""

    values = _matrix("expectations", expectations)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if n_shots is None:
        return values.copy()
    return sample_pauli_expectations(values, int(n_shots), rng=rng)


__all__ = [
    "EmulationMetrics",
    "MultiOutputRidge",
    "PairedWindowCertificate",
    "RIDGE_LSTSQ_RCOND",
    "TailSuffixCertificate",
    "emulation_metrics",
    "fit_multivariate_ridge",
    "paired_window_certificate",
    "row_rms",
    "sample_feature_matrix",
    "suffix_diameter",
    "tail_suffix_certificate",
    "window_lower_bound",
]
