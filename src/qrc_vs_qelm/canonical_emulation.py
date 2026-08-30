"""Canonical diagnostics for finite-window QRC-to-QELM emulation.

The central object is a suffix ensemble.  Every group contains several QRC
histories with an identical final window and independently sampled prefixes.
Any deterministic window model must make one prediction for the whole group.
This gives an exact finite-sample decomposition

    total squared error = history ambiguity + representation error.

The module intentionally contains no paper-specific decision thresholds.  It
provides deterministic primitives that runners can preregister and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log

import numpy as np


def _matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _suffix_cube(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape (suffix_groups, prefix_branches, outputs)"
        )
    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} dimensions must all be nonzero")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class SuffixRiskDecomposition:
    """Exact empirical risk decomposition on a grouped suffix ensemble."""

    history_mse: float
    representation_mse: float
    total_mse: float
    target_variance: float
    identity_error: float
    n_suffix_groups: int
    branches_per_group: int
    n_outputs: int

    @property
    def history_nrmse(self) -> float:
        return _normalized_root(self.history_mse, self.target_variance)

    @property
    def representation_nrmse(self) -> float:
        return _normalized_root(self.representation_mse, self.target_variance)

    @property
    def total_nrmse(self) -> float:
        return _normalized_root(self.total_mse, self.target_variance)

    @property
    def history_fraction(self) -> float:
        if self.total_mse <= 0.0:
            return 0.0
        return float(self.history_mse / self.total_mse)


def _normalized_root(value: float, scale: float) -> float:
    if scale <= 0.0:
        return float("inf") if value > 0.0 else 0.0
    return float(np.sqrt(max(float(value), 0.0) / float(scale)))


def suffix_risk_decomposition(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> SuffixRiskDecomposition:
    """Decompose held-out window-model error exactly.

    Parameters
    ----------
    targets:
        Array with shape ``(G, K, P)``.  The ``K`` target rows in a group come
        from different prefixes followed by the same length-``d`` suffix.
    predictions:
        One deterministic window-model prediction per suffix, shape ``(G, P)``.

    The biased within-group variance is deliberate: with that convention the
    empirical Pythagorean identity is exact, including for finite ``K``.
    """

    y = _suffix_cube("targets", targets)
    yhat = _matrix("predictions", predictions)
    if yhat.shape != (y.shape[0], y.shape[2]):
        raise ValueError(
            "predictions shape must equal (suffix_groups, outputs); "
            f"expected {(y.shape[0], y.shape[2])}, got {yhat.shape}"
        )

    group_mean = np.mean(y, axis=1)
    history_mse = float(np.mean((y - group_mean[:, None, :]) ** 2))
    representation_mse = float(np.mean((group_mean - yhat) ** 2))
    total_mse = float(np.mean((y - yhat[:, None, :]) ** 2))
    global_mean = np.mean(y, axis=(0, 1), keepdims=True)
    target_variance = float(np.mean((y - global_mean) ** 2))
    identity_error = float(total_mse - history_mse - representation_mse)

    return SuffixRiskDecomposition(
        history_mse=history_mse,
        representation_mse=representation_mse,
        total_mse=total_mse,
        target_variance=target_variance,
        identity_error=identity_error,
        n_suffix_groups=int(y.shape[0]),
        branches_per_group=int(y.shape[1]),
        n_outputs=int(y.shape[2]),
    )


def classify_emulation_regime(
    *,
    history_nrmse: float,
    student_nrmse: float,
    oracle_nrmse: float,
    target_variance: float,
    tolerance: float,
    min_target_variance: float = 1e-8,
) -> str:
    """Return a transparent possible/impossible category for one resource cell."""

    if not np.isfinite(float(target_variance)) or not np.isfinite(float(tolerance)):
        raise ValueError("target variance and tolerance must be finite")
    if not np.isfinite(float(min_target_variance)):
        raise ValueError("min_target_variance must be finite")
    if tolerance < 0.0 or min_target_variance < 0.0:
        raise ValueError("tolerances must be non-negative")

    if target_variance < min_target_variance:
        return "DEGENERATE_TARGET"
    if any(
        not np.isfinite(float(value))
        for value in (history_nrmse, student_nrmse, oracle_nrmse)
    ):
        raise ValueError("live-target NRMSE inputs must be finite")
    if history_nrmse > tolerance:
        return "HISTORY_IMPOSSIBLE"
    if student_nrmse <= tolerance:
        return "EMULATED"
    if oracle_nrmse <= tolerance:
        return "REPRESENTATION_LIMITED"
    return "UNRESOLVED"


@dataclass(frozen=True)
class ProjectionDiagnostic:
    """Linear-readout span diagnostics for one feature/target matrix."""

    rank: int
    augmented_width: int
    residual_mse: float
    target_variance: float
    target_aligned_fraction: float


def target_projection_diagnostic(
    features: np.ndarray,
    targets: np.ndarray,
) -> ProjectionDiagnostic:
    """Measure the best in-sample linear span available to a readout.

    This is a representation diagnostic, not a held-out performance estimate.
    It should be paired with independent test error in scientific reports.
    """

    x = _matrix("features", features)
    y = _matrix("targets", targets)
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must have the same number of rows")

    design = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
    fitted = design @ np.linalg.pinv(design) @ y
    residual_mse = float(np.mean((y - fitted) ** 2))
    centered = y - np.mean(y, axis=0, keepdims=True)
    target_variance = float(np.mean(centered**2))
    if target_variance <= 0.0:
        aligned = 1.0 if residual_mse <= 1e-15 else 0.0
    else:
        aligned = float(np.clip(1.0 - residual_mse / target_variance, 0.0, 1.0))
    return ProjectionDiagnostic(
        rank=int(np.linalg.matrix_rank(design)),
        augmented_width=int(design.shape[1]),
        residual_mse=residual_mse,
        target_variance=target_variance,
        target_aligned_fraction=aligned,
    )


def nested_snapshot_columns(
    *,
    max_snapshots: int,
    selected_snapshots: int,
    n_observables: int,
) -> np.ndarray:
    """Columns for a power-compatible snapshot subset that always includes ``T``.

    For example, a four-snapshot feature matrix has blocks at
    ``T/4, T/2, 3T/4, T``. Selecting two snapshots returns the ``T/2`` and ``T``
    blocks; selecting one returns only ``T``. Thus the budgets are truly nested.
    """

    max_snapshots = int(max_snapshots)
    selected_snapshots = int(selected_snapshots)
    n_observables = int(n_observables)
    if min(max_snapshots, selected_snapshots, n_observables) < 1:
        raise ValueError("snapshot counts and n_observables must be positive")
    if selected_snapshots > max_snapshots:
        raise ValueError("selected_snapshots must not exceed max_snapshots")
    if max_snapshots % selected_snapshots != 0:
        raise ValueError("selected_snapshots must divide max_snapshots")

    stride = max_snapshots // selected_snapshots
    blocks = np.arange(stride - 1, max_snapshots, stride, dtype=int)
    columns = [
        np.arange(block * n_observables, (block + 1) * n_observables, dtype=int)
        for block in blocks
    ]
    return np.concatenate(columns)


def select_nested_snapshots(
    features: np.ndarray,
    *,
    max_snapshots: int,
    selected_snapshots: int,
    n_observables: int,
) -> np.ndarray:
    """Select one nested measurement budget from a max-snapshot feature matrix."""

    x = _matrix("features", features)
    expected = int(max_snapshots) * int(n_observables)
    if x.shape[1] != expected:
        raise ValueError(
            f"features width must equal max_snapshots*n_observables={expected}"
        )
    columns = nested_snapshot_columns(
        max_snapshots=max_snapshots,
        selected_snapshots=selected_snapshots,
        n_observables=n_observables,
    )
    return x[:, columns]


@dataclass(frozen=True)
class RandomTanhFeatureMap:
    """A fixed classical ELM feature map fitted without target access."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray

    @property
    def width(self) -> int:
        return int(self.weights.shape[1])

    def transform(self, windows: np.ndarray, *, width: int | None = None) -> np.ndarray:
        x = _matrix("windows", windows)
        if x.shape[1] != self.weights.shape[0]:
            raise ValueError("window width does not match the fitted feature map")
        selected = self.width if width is None else int(width)
        if selected < 1 or selected > self.width:
            raise ValueError("width must satisfy 1 <= width <= fitted width")
        standardized = (x - self.mean) / self.scale
        return np.tanh(
            standardized @ self.weights[:, :selected] + self.bias[:selected]
        )


def fit_random_tanh_feature_map(
    train_windows: np.ndarray,
    *,
    width: int,
    seed: int,
    weight_scale: float = 1.0,
) -> RandomTanhFeatureMap:
    """Fit train-only input scaling and draw one nested classical ELM bank."""

    x = _matrix("train_windows", train_windows)
    width = int(width)
    if width < 1:
        raise ValueError("width must be positive")
    if not np.isfinite(float(weight_scale)) or float(weight_scale) <= 0.0:
        raise ValueError("weight_scale must be finite and positive")

    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    rng = np.random.default_rng(int(seed))
    weights = rng.normal(
        0.0,
        float(weight_scale) / np.sqrt(x.shape[1]),
        size=(x.shape[1], width),
    )
    bias = rng.uniform(-1.0, 1.0, size=width)
    return RandomTanhFeatureMap(
        mean=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
    )


def degree_two_delay_features(windows: np.ndarray) -> np.ndarray:
    """Return linear and unique quadratic monomials of each delay window."""

    x = _matrix("windows", windows)
    terms = [x]
    terms.extend(
        (x[:, i] * x[:, j])[:, None]
        for i in range(x.shape[1])
        for j in range(i, x.shape[1])
    )
    return np.column_stack(terms)


@dataclass(frozen=True)
class ExponentialForgettingFit:
    """Log-linear empirical forgetting fit ``witness(d)=amplitude*exp(-d/tau)``."""

    amplitude: float
    tau: float
    slope: float
    r_squared: float
    n_points: int

    def predicted_window(self, tolerance: float) -> int | None:
        """Smallest positive integer window predicted below the tolerance."""

        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            return None
        if self.amplitude <= tolerance:
            return 1
        return int(max(1, ceil(self.tau * log(self.amplitude / tolerance))))


def fit_exponential_forgetting(
    windows: np.ndarray,
    witness: np.ndarray,
    *,
    min_witness: float = 1e-14,
) -> ExponentialForgettingFit:
    """Fit an elementary exponential prediction to positive witness values."""

    d = np.asarray(windows, dtype=np.float64)
    values = np.asarray(witness, dtype=np.float64)
    if d.ndim != 1 or values.ndim != 1 or d.shape != values.shape:
        raise ValueError("windows and witness must be same-length vectors")
    if d.size < 2 or not np.all(np.isfinite(d)) or not np.all(np.isfinite(values)):
        raise ValueError("windows and witness need at least two finite points")
    if np.any(d < 0.0) or float(min_witness) <= 0.0:
        raise ValueError("windows must be non-negative and min_witness positive")

    mask = values > float(min_witness)
    if int(np.sum(mask)) < 2:
        raise ValueError("at least two witness values must exceed min_witness")
    x = d[mask]
    y = np.log(values[mask])
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ np.array([intercept, slope])
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if ss_tot <= 0.0 and ss_res <= 1e-15 else 1.0 - ss_res / max(
        ss_tot, 1e-30
    )
    tau = float("inf") if slope >= 0.0 else float(-1.0 / slope)
    return ExponentialForgettingFit(
        amplitude=float(np.exp(intercept)),
        tau=tau,
        slope=float(slope),
        r_squared=float(r_squared),
        n_points=int(len(x)),
    )
