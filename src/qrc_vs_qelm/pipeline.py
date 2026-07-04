"""Shared train/validation/test ridge-readout evaluation pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qrc_vs_qelm.core import readout
from qrc_vs_qelm.matching import participation_effective_rank
from qrc_vs_qelm.models.api import FeatureBundle


_DEFAULT_ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2)


@dataclass(frozen=True)
class Splits:
    """Full-length row indices for contiguous train/val/test regions."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def all(self) -> np.ndarray:
        return np.concatenate([self.train, self.val, self.test])


@dataclass(frozen=True)
class EvaluationContext:
    """Exact rows and target values consumed by one evaluation call."""

    indices: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    targets: dict[str, np.ndarray]


def _as_index_vector(name: str, values: np.ndarray) -> np.ndarray:
    idx = np.asarray(values, dtype=int)
    if idx.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional index array")
    return idx


def _as_nonnegative_int(name: str, value: int) -> int:
    checked = int(value)
    if checked != value or checked < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return checked


def _as_target_dict(targets: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    checked: dict[str, np.ndarray] = {}
    for name, y in targets.items():
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"target {name!r} must be one-dimensional")
        checked[str(name)] = arr
    if not checked:
        raise ValueError("targets must contain at least one target")
    return checked


def valid_indices(
    T: int,
    washout: int,
    max_delay: int,
    targets: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Return the single shared valid row set for a comparison.

    Rows satisfy ``t >= max(washout, max_delay)``.  When targets are provided,
    any row with a non-finite value in any target is removed from the shared set.
    """

    T = int(T)
    washout = int(washout)
    max_delay = int(max_delay)
    if T < 0:
        raise ValueError("T must be non-negative")
    if washout < 0:
        raise ValueError("washout must be non-negative")
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative")

    idx = np.arange(max(washout, max_delay), T, dtype=int)
    if targets is None:
        return idx

    mask = np.ones(T, dtype=bool)
    for name, y in _as_target_dict(targets).items():
        if y.shape[0] != T:
            raise ValueError(
                f"target {name!r} length {y.shape[0]} does not match T={T}"
            )
        mask &= np.isfinite(y)
    return idx[mask[idx]]


def stage_valid_indices(
    bundles: Sequence[FeatureBundle],
    targets: Mapping[str, np.ndarray],
    washout: int,
    max_delay: int = 0,
) -> np.ndarray:
    """Return shared runner-valid rows including bundle warm-up cutoffs."""

    checked = _as_target_dict(targets)
    lengths = {y.shape[0] for y in checked.values()}
    if len(lengths) != 1:
        raise ValueError("all targets must have the same length")
    T = lengths.pop()

    effective_max_delay = _as_nonnegative_int("max_delay", max_delay)
    for pos, bundle in enumerate(bundles):
        first_valid = _as_nonnegative_int(
            f"bundles[{pos}].first_valid",
            bundle.first_valid,
        )
        features = np.asarray(bundle.features)
        if features.ndim != 2:
            raise ValueError(f"bundles[{pos}].features must be two-dimensional")
        if features.shape[0] != T:
            raise ValueError(
                f"bundles[{pos}].features rows {features.shape[0]} do not "
                f"match target length {T}"
            )
        effective_max_delay = max(effective_max_delay, first_valid)

    return valid_indices(T, washout, effective_max_delay, checked)


def make_splits(
    idx: np.ndarray,
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
) -> Splits:
    """Make ordered contiguous train/val/test splits over valid row indices."""

    ordered = _as_index_vector("idx", idx)
    if ordered.size < 3:
        raise ValueError("at least three valid rows are required")
    if np.any(np.diff(ordered) <= 0):
        raise ValueError("idx must be strictly increasing")

    fracs = tuple(float(f) for f in fractions)
    if len(fracs) != 3:
        raise ValueError("fractions must contain exactly three entries")
    if any(f <= 0.0 for f in fracs):
        raise ValueError("split fractions must be positive")
    total = sum(fracs)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("split fractions must have positive finite sum")

    n = ordered.size
    n_train = int(np.floor(n * fracs[0] / total))
    n_val = int(np.floor(n * fracs[1] / total))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)

    return Splits(
        train=ordered[:n_train].copy(),
        val=ordered[n_train : n_train + n_val].copy(),
        test=ordered[n_train + n_val :].copy(),
    )


def standardize_train_only(X: np.ndarray, splits: Splits) -> np.ndarray:
    """Standardize columns using train rows only, guarding constant columns."""

    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    train = _as_index_vector("splits.train", splits.train)
    if train.size == 0:
        raise ValueError("train split must not be empty")
    if train.min() < 0 or train.max() >= values.shape[0]:
        raise ValueError("train split contains row indices outside X")

    mu = values[train].mean(axis=0)
    sigma = values[train].std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return ((values - mu) / sigma).astype(np.float64, copy=False)


def prepare_evaluation_context(
    targets: Mapping[str, np.ndarray],
    splits: Splits,
) -> EvaluationContext:
    """Return the exact target slices and split rows used for scoring."""

    checked = _as_target_dict(targets)
    train = _as_index_vector("splits.train", splits.train).copy()
    val = _as_index_vector("splits.val", splits.val).copy()
    test = _as_index_vector("splits.test", splits.test).copy()
    indices = np.concatenate([train, val, test])
    sliced_targets = {name: y[indices].copy() for name, y in checked.items()}
    return EvaluationContext(
        indices=indices.copy(),
        train_idx=train,
        val_idx=val,
        test_idx=test,
        targets=sliced_targets,
    )


def _effective_rank(X_train: np.ndarray) -> float:
    # Report the matching participation-ratio rank; core keeps its entropy rank.
    return float(participation_effective_rank(X_train))


def _fit_centered_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Fit ridge with an unpenalized intercept from training rows only."""

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X_train must be a two-dimensional array")
    if y.ndim != 1:
        raise ValueError("y_train must be a one-dimensional array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X_train and y_train must have the same number of rows")
    if X.shape[0] == 0:
        raise ValueError("X_train must contain at least one row")

    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean
    alpha = float(alpha)
    if alpha <= 0.0:
        weights, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    else:
        n_feat = X.shape[1]
        A = Xc.T @ Xc + alpha * np.eye(n_feat)
        b = Xc.T @ yc
        weights = np.linalg.solve(A, b)
    intercept = y_mean - float(x_mean @ weights)
    return weights, intercept


def _fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    weights, intercept = _fit_centered_ridge(X_train, y_train, alpha)
    return np.asarray(X_eval, dtype=np.float64) @ weights + intercept


def _validate_split_rows(
    X: np.ndarray,
    targets: Mapping[str, np.ndarray],
    splits: Splits,
    first_valid: int = 0,
) -> None:
    rows = splits.all
    if rows.size == 0:
        raise ValueError("splits must contain at least one row")
    if rows.min() < 0 or rows.max() >= X.shape[0]:
        raise ValueError("splits contain row indices outside X")
    if np.unique(rows).size != rows.size:
        raise ValueError("splits must be disjoint")
    first_valid = _as_nonnegative_int("first_valid", first_valid)
    if rows.min() < first_valid:
        bad_regions = []
        for region_name, region_rows in (
            ("train", splits.train),
            ("val", splits.val),
            ("test", splits.test),
        ):
            if region_rows.size and region_rows.min() < first_valid:
                bad_regions.append(region_name)
        regions = ", ".join(bad_regions)
        raise ValueError(
            f"splits contain rows below first_valid={first_valid} "
            f"in {regions}"
        )

    for region_name, region_rows in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        region = X[region_rows]
        bad = ~np.isfinite(region)
        if bad.any():
            local_row, col = np.argwhere(bad)[0]
            row = int(region_rows[local_row])
            raise ValueError(
                f"X has non-finite values in {region_name} split "
                f"at row {row}, column {int(col)}"
            )

    for name, y in targets.items():
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"target {name!r} length {y.shape[0]} does not match X rows "
                f"{X.shape[0]}"
            )
        if not np.isfinite(y[rows]).all():
            raise ValueError(
                f"target {name!r} has non-finite values inside the shared splits"
            )


def evaluate_feature_matrix(
    X: np.ndarray,
    targets: Mapping[str, np.ndarray],
    splits: Splits,
    alphas: Sequence[float] = _DEFAULT_ALPHAS,
    first_valid: int = 0,
) -> pd.DataFrame:
    """Evaluate all targets with the same standardized features and split rows."""

    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    checked_targets = _as_target_dict(targets)
    _validate_split_rows(values, checked_targets, splits, first_valid=first_valid)

    alpha_grid = tuple(float(alpha) for alpha in alphas)
    if not alpha_grid:
        raise ValueError("alphas must not be empty")
    if any(alpha < 0.0 or not np.isfinite(alpha) for alpha in alpha_grid):
        raise ValueError("alphas must be finite non-negative values")

    Xs = standardize_train_only(values, splits)
    train = splits.train
    val = splits.val
    test = splits.test
    train_val = np.concatenate([train, val])
    rank = _effective_rank(Xs[train])

    rows: list[dict[str, float | int | str]] = []
    for name, y in checked_targets.items():
        best_alpha = alpha_grid[0]
        best_capacity = -np.inf
        for alpha in alpha_grid:
            pred_val = _fit_predict(Xs[train], y[train], Xs[val], alpha)
            capacity_val = readout.capacity(y[val], pred_val)
            if capacity_val > best_capacity:
                best_capacity = capacity_val
                best_alpha = alpha

        pred_test = _fit_predict(
            Xs[train_val],
            y[train_val],
            Xs[test],
            best_alpha,
        )
        rows.append(
            {
                "target": name,
                "alpha": float(best_alpha),
                "capacity_test": readout.capacity(y[test], pred_test),
                "nrmse_test": readout.nrmse(y[test], pred_test),
                "n_train": int(train.size),
                "n_val": int(val.size),
                "n_test": int(test.size),
                "effective_rank": rank,
                "feature_dim": int(values.shape[1]),
            }
        )

    columns = [
        "target",
        "alpha",
        "capacity_test",
        "nrmse_test",
        "n_train",
        "n_val",
        "n_test",
        "effective_rank",
        "feature_dim",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.attrs["evaluation_context"] = prepare_evaluation_context(
        checked_targets,
        splits,
    )
    return df
