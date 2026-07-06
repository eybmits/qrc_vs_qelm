"""Information processing capacity estimators for Dambre-basis tasks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import itertools
import logging

import numpy as np

from qrc_vs_qelm import pipeline, task_families


LOGGER = logging.getLogger(__name__)
DEFAULT_N_SURROGATES = 32
DEFAULT_CUMULATIVE_SURROGATE_EPSILON = 1e-12
DEFAULT_BASIS_EDGE_WARNING_FRACTION = 0.05


@dataclass(frozen=True)
class BasisSpec:
    """One Dambre basis function specification."""

    factors: tuple[tuple[int, int], ...]
    total_degree: int
    key: str

    def __post_init__(self) -> None:
        factors = tuple((int(tau), int(degree)) for tau, degree in self.factors)
        if not factors:
            raise ValueError("BasisSpec.factors must not be empty")
        if any(tau < 0 for tau, _ in factors):
            raise ValueError("BasisSpec delays must be non-negative")
        if any(degree < 1 for _, degree in factors):
            raise ValueError("BasisSpec degrees must be positive")
        delays = tuple(tau for tau, _ in factors)
        if len(set(delays)) != len(delays):
            raise ValueError("BasisSpec delays must be distinct")
        total = sum(degree for _, degree in factors)
        if int(self.total_degree) != total:
            raise ValueError("BasisSpec.total_degree must equal the factor sum")
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "total_degree", total)
        object.__setattr__(self, "key", str(self.key))


class Basis(list[BasisSpec]):
    """List-like basis container carrying truncation metadata."""

    truncated: bool

    def __init__(
        self,
        specs: Iterable[BasisSpec] = (),
        *,
        truncated: bool = False,
    ) -> None:
        super().__init__(specs)
        self.truncated = bool(truncated)


def _nonnegative_int(name: str, value: int) -> int:
    checked = int(value)
    if checked != value or checked < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return checked


def _positive_int(name: str, value: int) -> int:
    checked = int(value)
    if checked != value or checked < 1:
        raise ValueError(f"{name} must be a positive integer")
    return checked


def _as_1d_float(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return arr


def _as_2d_float(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return arr


def _rng_from(rng: np.random.Generator | int) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


def _degree_compositions(
    total: int,
    n_parts: int,
    max_part: int,
) -> Iterable[tuple[int, ...]]:
    if n_parts == 1:
        if 1 <= total <= max_part:
            yield (total,)
        return

    min_rest = n_parts - 1
    max_first = min(max_part, total - min_rest)
    for first in range(1, max_first + 1):
        for rest in _degree_compositions(total - first, n_parts - 1, max_part):
            yield (first, *rest)


def _basis_key(factors: Sequence[tuple[int, int]], total_degree: int) -> str:
    parts = "_".join(f"tau{tau}_p{degree}" for tau, degree in factors)
    return f"ipc_d{total_degree}_{parts}"


def dambre_basis(
    max_delay: int,
    max_degree: int,
    max_total_degree: int,
    max_functions: int,
) -> list[BasisSpec]:
    """Enumerate Dambre Legendre-product basis functions in stable order."""

    max_delay = _nonnegative_int("max_delay", max_delay)
    max_degree = _positive_int("max_degree", max_degree)
    max_total_degree = _positive_int("max_total_degree", max_total_degree)
    max_functions = _nonnegative_int("max_functions", max_functions)

    delays = tuple(range(max_delay + 1))
    specs: list[BasisSpec] = []
    truncated = False

    for total_degree in range(1, max_total_degree + 1):
        max_factors = min(total_degree, len(delays))
        for n_factors in range(1, max_factors + 1):
            for delay_combo in itertools.combinations(delays, n_factors):
                for degree_combo in _degree_compositions(
                    total_degree,
                    n_factors,
                    max_degree,
                ):
                    factors = tuple(zip(delay_combo, degree_combo, strict=True))
                    spec = BasisSpec(
                        factors=factors,
                        total_degree=total_degree,
                        key=_basis_key(factors, total_degree),
                    )
                    if len(specs) >= max_functions:
                        LOGGER.warning(
                            "Dambre basis truncated at max_functions=%d; "
                            "at least one basis function was omitted",
                            max_functions,
                        )
                        truncated = True
                        return Basis(specs, truncated=truncated)
                    specs.append(spec)

    return Basis(specs, truncated=truncated)


def basis_target(u: np.ndarray, spec: BasisSpec) -> np.ndarray:
    """Build the Legendre-product target for one basis specification."""

    if not isinstance(spec, BasisSpec):
        raise TypeError("spec must be a BasisSpec")
    values = _as_1d_float("u", u)
    target = np.ones(values.shape[0], dtype=np.float64)
    for tau, degree in spec.factors:
        target *= task_families.legendre_target(values, tau=tau, degree=degree)
    return target


def _basis_targets(
    u: np.ndarray,
    basis: Sequence[BasisSpec],
) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for spec in basis:
        if spec.key in targets:
            raise ValueError(f"duplicate basis key: {spec.key!r}")
        targets[spec.key] = basis_target(u, spec)
    return targets


def _capacities_by_key(
    features: np.ndarray,
    targets: Mapping[str, np.ndarray],
    splits: pipeline.Splits,
    alphas: Sequence[float],
) -> dict[str, float]:
    if not targets:
        return {}
    frame = pipeline.evaluate_feature_matrix(features, targets, splits, alphas=alphas)
    return {
        str(row.target): float(np.clip(row.capacity_test, 0.0, 1.0))
        for row in frame.itertuples(index=False)
    }


def _spec_max_delay(spec: BasisSpec) -> int:
    return max(tau for tau, _ in spec.factors)


def _basis_degrees(basis: Sequence[BasisSpec]) -> tuple[int, ...]:
    return tuple(sorted({spec.total_degree for spec in basis}))


def _thresholds_from_scalar(
    basis: Sequence[BasisSpec],
    threshold: float,
) -> dict[int, float]:
    return {degree: float(threshold) for degree in _basis_degrees(basis)}


def _retained_surrogate_total(
    capacities: Mapping[str, float],
    basis: Sequence[BasisSpec],
    thresholds_by_degree: Mapping[int, float],
) -> float:
    retained = 0.0
    for spec in basis:
        capacity = float(capacities.get(spec.key, 0.0))
        threshold = float(thresholds_by_degree.get(spec.total_degree, np.inf))
        if capacity > threshold:
            retained += capacity
    return float(retained)


def _raise_thresholds_for_cumulative_noise(
    surrogate_capacities: Sequence[Mapping[str, float]],
    basis: Sequence[BasisSpec],
    thresholds_by_degree: dict[int, float],
    epsilon: float,
) -> dict[int, float]:
    """Raise per-degree thresholds until every surrogate's retained total is tiny."""

    if not surrogate_capacities:
        return dict(thresholds_by_degree)

    thresholds = dict(thresholds_by_degree)
    epsilon = float(epsilon)
    if epsilon < 0.0 or not np.isfinite(epsilon):
        raise ValueError("cumulative_surrogate_epsilon must be finite and non-negative")

    for capacities in surrogate_capacities:
        while _retained_surrogate_total(capacities, basis, thresholds) > epsilon:
            retained = [
                (float(capacities.get(spec.key, 0.0)), spec.total_degree)
                for spec in basis
                if float(capacities.get(spec.key, 0.0))
                > float(thresholds.get(spec.total_degree, np.inf))
            ]
            if not retained:
                break
            capacity, degree = max(retained, key=lambda item: item[0])
            thresholds[degree] = max(float(thresholds.get(degree, 0.0)), capacity)

    return thresholds


def _surrogate_threshold_diagnostics(
    surrogate_capacities: Sequence[Mapping[str, float]],
    basis: Sequence[BasisSpec],
    thresholds_by_degree: Mapping[int, float],
) -> dict[str, float]:
    retained_totals = [
        _retained_surrogate_total(capacities, basis, thresholds_by_degree)
        for capacities in surrogate_capacities
    ]
    if not retained_totals:
        return {
            "surrogate_retained_capacity_max": 0.0,
            "surrogate_retained_capacity_mean": 0.0,
        }
    return {
        "surrogate_retained_capacity_max": float(np.max(retained_totals)),
        "surrogate_retained_capacity_mean": float(np.mean(retained_totals)),
    }


def significance_thresholds_by_degree(
    features: np.ndarray,
    u: np.ndarray,
    basis: Sequence[BasisSpec],
    splits: pipeline.Splits,
    alphas: Sequence[float],
    *,
    n_surrogates: int,
    rng: np.random.Generator | int,
    cumulative_surrogate_epsilon: float = DEFAULT_CUMULATIVE_SURROGATE_EPSILON,
) -> tuple[dict[int, float], dict[str, float]]:
    """Estimate per-degree capacity floors from shuffled-input surrogates.

    The floor for each degree is at least the largest same-degree surrogate
    component and the largest same-degree surrogate cumulative capacity. The
    latter is deliberately conservative: it prevents many moderate shuffled
    capacities in one degree class from being retained together.
    """

    values = _as_1d_float("u", u)
    X = _as_2d_float("features", features)
    if X.shape[0] != values.shape[0]:
        raise ValueError("features and u must have the same number of rows")
    n_surrogates = _positive_int("n_surrogates", n_surrogates)
    if len(basis) == 0:
        return {}, {
            "surrogate_retained_capacity_max": 0.0,
            "surrogate_retained_capacity_mean": 0.0,
        }

    generator = _rng_from(rng)
    by_degree: dict[int, list[float]] = {degree: [] for degree in _basis_degrees(basis)}
    totals_by_degree: dict[int, list[float]] = {
        degree: [] for degree in _basis_degrees(basis)
    }
    surrogate_capacities: list[dict[str, float]] = []
    for _ in range(n_surrogates):
        shuffled = values[generator.permutation(values.shape[0])]
        targets = _basis_targets(shuffled, basis)
        capacities = _capacities_by_key(X, targets, splits, alphas)
        surrogate_capacities.append(capacities)
        degree_totals = {degree: 0.0 for degree in totals_by_degree}
        for spec in basis:
            capacity = float(capacities.get(spec.key, 0.0))
            by_degree[spec.total_degree].append(capacity)
            degree_totals[spec.total_degree] += capacity
        for degree, total in degree_totals.items():
            totals_by_degree[degree].append(total)

    thresholds = {
        degree: float(
            max(
                max(capacities, default=0.0),
                max(totals_by_degree.get(degree, ()), default=0.0),
            )
        )
        for degree, capacities in by_degree.items()
    }
    thresholds = _raise_thresholds_for_cumulative_noise(
        surrogate_capacities,
        basis,
        thresholds,
        cumulative_surrogate_epsilon,
    )
    diagnostics = _surrogate_threshold_diagnostics(
        surrogate_capacities,
        basis,
        thresholds,
    )
    return thresholds, diagnostics


def significance_threshold(
    features: np.ndarray,
    u: np.ndarray,
    basis: Sequence[BasisSpec],
    splits: pipeline.Splits,
    alphas: Sequence[float],
    *,
    n_surrogates: int,
    rng: np.random.Generator | int,
) -> float:
    """Return the largest per-degree shuffled-input capacity floor.

    ``total_ipc`` uses the full per-degree thresholds. This scalar helper remains
    for callers that need a single conservative diagnostic value.
    """

    thresholds, _ = significance_thresholds_by_degree(
        features,
        u,
        basis,
        splits,
        alphas,
        n_surrogates=n_surrogates,
        rng=rng,
    )
    return float(max(thresholds.values(), default=0.0))


def _centered_feature_rank(features: np.ndarray, splits: pipeline.Splits) -> int:
    rows = splits.all
    if rows.size == 0:
        return 0
    selected = np.asarray(features[rows], dtype=np.float64)
    centered = selected - selected.mean(axis=0, keepdims=True)
    return int(np.linalg.matrix_rank(centered))


def _scale_summary(
    values: Mapping[int, float],
    scale: float,
) -> dict[int, float]:
    return {key: float(value * scale) for key, value in values.items()}


def total_ipc(
    features: np.ndarray,
    u: np.ndarray,
    basis: Sequence[BasisSpec],
    splits: pipeline.Splits,
    alphas: Sequence[float],
    *,
    threshold: float | None = None,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    cumulative_surrogate_epsilon: float = DEFAULT_CUMULATIVE_SURROGATE_EPSILON,
    rng: np.random.Generator | int,
) -> dict[str, object]:
    """Compute thresholded total IPC and degree/delay IPC summaries.

    With ``threshold=None``, shuffled-input surrogates select deterministic
    per-degree thresholds and verify that the retained shuffled total is no more
    than ``cumulative_surrogate_epsilon``. For a fair QRC-vs-QELM delta, the
    caller must pass the same ``basis``, ``alphas``, ``n_surrogates``, and
    threshold-selection protocol to both models. If QRC and TD-QELM(k) have
    different warm-up rows, evaluate all models on one shared valid-index set:
    the intersection implied by the largest ``first_valid`` in the k-sweep.
    """

    values = _as_1d_float("u", u)
    X = _as_2d_float("features", features)
    if X.shape[0] != values.shape[0]:
        raise ValueError("features and u must have the same number of rows")

    threshold_diagnostics = {
        "surrogate_retained_capacity_max": 0.0,
        "surrogate_retained_capacity_mean": 0.0,
    }
    if threshold is None:
        thresholds_by_degree, threshold_diagnostics = significance_thresholds_by_degree(
            X,
            values,
            basis,
            splits,
            alphas,
            n_surrogates=n_surrogates,
            cumulative_surrogate_epsilon=cumulative_surrogate_epsilon,
            rng=rng,
        )
        threshold_value = float(max(thresholds_by_degree.values(), default=0.0))
    else:
        threshold_value = float(threshold)
        if threshold_value < 0.0 or not np.isfinite(threshold_value):
            raise ValueError("threshold must be a finite non-negative value")
        thresholds_by_degree = _thresholds_from_scalar(basis, threshold_value)

    targets = _basis_targets(values, basis)
    capacities = _capacities_by_key(X, targets, splits, alphas)

    raw_by_degree: dict[int, float] = {degree: 0.0 for degree in _basis_degrees(basis)}
    raw_by_delay: dict[int, float] = {
        delay: 0.0 for delay in sorted({_spec_max_delay(spec) for spec in basis})
    }
    n_significant = 0
    max_basis_delay = max((_spec_max_delay(spec) for spec in basis), default=0)
    max_basis_degree = max((spec.total_degree for spec in basis), default=0)
    basis_edge_capacity = 0.0
    capacities_by_key: dict[str, float] = {}
    retained_capacities_by_key: dict[str, float] = {}
    for spec in basis:
        capacity = float(capacities.get(spec.key, 0.0))
        capacities_by_key[spec.key] = capacity
        degree_threshold = thresholds_by_degree.get(spec.total_degree, threshold_value)
        if capacity <= degree_threshold:
            capacity = 0.0
        else:
            n_significant += 1
        retained_capacities_by_key[spec.key] = capacity
        raw_by_degree[spec.total_degree] = raw_by_degree.get(spec.total_degree, 0.0) + capacity
        delay = _spec_max_delay(spec)
        raw_by_delay[delay] = raw_by_delay.get(delay, 0.0) + capacity
        if delay == max_basis_delay or spec.total_degree == max_basis_degree:
            basis_edge_capacity += capacity

    raw_by_degree = {degree: raw_by_degree[degree] for degree in sorted(raw_by_degree)}
    raw_by_delay = {delay: raw_by_delay[delay] for delay in sorted(raw_by_delay)}
    raw_total = float(sum(raw_by_degree.values()))
    effective_rank = _centered_feature_rank(X, splits)
    total = float(min(raw_total, float(effective_rank)))
    bound_excess = float(max(0.0, raw_total - float(effective_rank)))
    if bound_excess > 1e-12:
        LOGGER.warning(
            "Clamped total_ipc from raw_sum=%.12g to effective_rank=%d "
            "(bound_excess=%.12g)",
            raw_total,
            effective_rank,
            bound_excess,
        )

    scale = 0.0 if raw_total <= 0.0 else total / raw_total
    by_degree = _scale_summary(raw_by_degree, scale)
    by_delay = _scale_summary(raw_by_delay, scale)
    mc = float(by_degree.get(1, 0.0))
    nc = float(sum(value for degree, value in by_degree.items() if degree >= 2))
    if raw_total > 0.0:
        basis_edge_capacity = float(basis_edge_capacity / raw_total)
    else:
        basis_edge_capacity = 0.0
    if basis_edge_capacity > DEFAULT_BASIS_EDGE_WARNING_FRACTION:
        LOGGER.warning(
            "IPC basis edge capacity is %.3f at max_delay=%d/max_degree=%d; "
            "basis caps may under-count longer-memory or higher-degree capacity",
            basis_edge_capacity,
            max_basis_delay,
            max_basis_degree,
        )

    return {
        "total_ipc": total,
        "raw_total_ipc": raw_total,
        "raw_sum": raw_total,
        "effective_rank": int(effective_rank),
        "bound_excess": bound_excess,
        "mc": mc,
        "nc": nc,
        "n_significant": int(n_significant),
        "n_basis": int(len(basis)),
        "threshold": threshold_value,
        "thresholds_by_degree": {
            degree: float(thresholds_by_degree[degree])
            for degree in sorted(thresholds_by_degree)
        },
        **threshold_diagnostics,
        "truncated": bool(getattr(basis, "truncated", False)),
        "by_degree": by_degree,
        "by_delay": by_delay,
        "raw_by_degree": raw_by_degree,
        "raw_by_delay": raw_by_delay,
        "capacities_by_key": capacities_by_key,
        "retained_capacities_by_key": retained_capacities_by_key,
        "basis_edge_capacity": basis_edge_capacity,
    }


__all__ = [
    "Basis",
    "BasisSpec",
    "DEFAULT_N_SURROGATES",
    "DEFAULT_CUMULATIVE_SURROGATE_EPSILON",
    "basis_target",
    "dambre_basis",
    "significance_threshold",
    "significance_thresholds_by_degree",
    "total_ipc",
]
