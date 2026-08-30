"""Architecture-blind inference for recurrence-fingerprint trajectories.

The classifier in this module consumes already collected scores.  It does not
import a simulator, inspect an architecture name, or use a declared window.
Its conclusions are deliberately operational: a trajectory can be compatible
with an observed finite cutoff, can exhibit a reset-sensitive recurrent tail,
or can remain unresolved because its baseline is too small or its evidence is
otherwise inconclusive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np
import pandas as pd


INFINITY = "infinity"

GROUP_COLUMNS = ("blind_id", "shots_per_batch", "measurement_seed")
TRAJECTORY_COLUMNS = ("physics_id", "history_id")
REQUIRED_COLUMNS = frozenset(
    {
        *GROUP_COLUMNS,
        *TRAJECTORY_COLUMNS,
        "condition",
        "suffix_length",
        "cross_batch_signal_mse",
    }
)

# These columns can reveal the answer or a privileged architecture parameter.
# Matching is case-insensitive and ignores surrounding whitespace.
FORBIDDEN_LABEL_COLUMNS = frozenset(
    {
        "actual_window",
        "adapter_type",
        "algorithm",
        "architecture",
        "architecture_id",
        "class",
        "classification",
        "curve_model",
        "declared_window",
        "encoding",
        "expected_behavior",
        "family",
        "fingerprint_label",
        "ground_truth",
        "is_qelm",
        "is_qrc",
        "label",
        "method",
        "model",
        "model_family",
        "model_id",
        "model_name",
        "qelm_window",
        "source_family",
        "source_model",
        "system_type",
        "target",
        "true_label",
        "window",
        "window_size",
    }
)


class FingerprintLabel(str, Enum):
    """Conservative labels returned by :func:`classify_blind`."""

    FINITE_WINDOW_COMPATIBLE = "FINITE_WINDOW_COMPATIBLE"
    RECURRENT_RESET_SENSITIVE = "RECURRENT_RESET_SENSITIVE"
    RECURRENT_RESET_INSENSITIVE = "RECURRENT_RESET_INSENSITIVE"
    INDETERMINATE_LOW_SIGNAL = "INDETERMINATE_LOW_SIGNAL"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class FingerprintRule:
    """Pre-specified thresholds for an architecture-blind fingerprint test.

    ``max_window`` is the largest cutoff that may be inferred.  A finite cutoff
    at ``k`` requires demonstrated signal at suffix ``k - 1`` as well as
    equivalence to zero at every registered suffix from ``k`` onward.

    A reset-sensitive recurrent label uses only ``tail_suffixes``: the mean carry
    tail must remain positive, the reset tail must be uniformly near zero, and
    their mean contrast must remain positive.  These labels describe the observed
    intervention only; they do not identify a model architecture.
    """

    max_window: int
    baseline_floor: float
    equivalence_ratio: float
    liveness_ratio: float
    recurrent_tail_ratio: float
    reset_max_ratio: float
    reset_contrast_ratio: float
    tail_suffixes: tuple[int, ...]
    confidence: float = 0.95
    carry_condition: str = "carry"
    reset_condition: str = "ideal_quantum_reset"

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_window, (bool, np.bool_))
            or int(self.max_window) != self.max_window
            or int(self.max_window) < 1
        ):
            raise ValueError("max_window must be an integer >= 1")
        for name in (
            "baseline_floor",
            "equivalence_ratio",
            "liveness_ratio",
            "recurrent_tail_ratio",
            "reset_max_ratio",
            "reset_contrast_ratio",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.liveness_ratio <= self.equivalence_ratio:
            raise ValueError("liveness_ratio must be greater than equivalence_ratio")
        if self.recurrent_tail_ratio <= self.equivalence_ratio:
            raise ValueError(
                "recurrent_tail_ratio must be greater than equivalence_ratio"
            )
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        tail_suffixes = tuple(int(value) for value in self.tail_suffixes)
        if tail_suffixes != tuple(sorted(set(tail_suffixes))):
            raise ValueError("tail_suffixes must be unique and sorted")
        if len(tail_suffixes) < 2:
            raise ValueError("tail_suffixes must contain at least two points")
        if any(value < int(self.max_window) for value in tail_suffixes):
            raise ValueError("tail_suffixes must be at or beyond max_window")
        if not isinstance(self.carry_condition, str) or not self.carry_condition:
            raise ValueError("carry_condition must be a non-empty string")
        if not isinstance(self.reset_condition, str) or not self.reset_condition:
            raise ValueError("reset_condition must be a non-empty string")
        if self.carry_condition == self.reset_condition:
            raise ValueError("carry and reset conditions must differ")

        object.__setattr__(self, "max_window", int(self.max_window))
        object.__setattr__(self, "tail_suffixes", tail_suffixes)


@dataclass(frozen=True)
class CrossBatchEstimate:
    """Shot-noise-aware squared-distance estimates from two sample batches."""

    signal_mse: float
    raw_two_batch_mse: float


def sample_pauli_expectations(
    expectations: np.ndarray | Iterable[float],
    n_shots: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample independent exact-binomial means of +/-1 Pauli outcomes."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if isinstance(n_shots, (bool, np.bool_)) or int(n_shots) != n_shots:
        raise ValueError("n_shots must be an integer >= 1")
    shots = int(n_shots)
    if shots < 1:
        raise ValueError("n_shots must be an integer >= 1")

    values = np.asarray(expectations, dtype=float)
    if values.size == 0:
        raise ValueError("expectations must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("expectations must be finite")
    tolerance = 16.0 * np.finfo(float).eps
    if np.any(values < -1.0 - tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError("Pauli expectations must lie in [-1, 1]")

    clipped = np.clip(values, -1.0, 1.0)
    plus_counts = rng.binomial(shots, 0.5 * (1.0 + clipped))
    return (2.0 * plus_counts / shots - 1.0).astype(float)


def estimate_cross_batch_signal_mse(
    expectations_a: np.ndarray | Iterable[float],
    expectations_b: np.ndarray | Iterable[float],
    shots_per_batch: int | None,
    *,
    rng: np.random.Generator,
) -> CrossBatchEstimate:
    """Estimate a squared feature distance without the shot-noise MSE bias.

    Two statistically independent sample batches are drawn for each arm.  The
    cross product of the two difference vectors is unbiased for the squared
    distance.  ``raw_two_batch_mse`` is returned as a diagnostic and retains
    the usual positive shot-noise bias.  Passing ``None`` selects the exact
    expectation-value tier.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    first = np.asarray(expectations_a, dtype=float)
    second = np.asarray(expectations_b, dtype=float)
    if first.shape != second.shape:
        raise ValueError("expectation arrays must have identical shapes")
    if first.size == 0:
        raise ValueError("expectation arrays must be non-empty")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("expectation arrays must be finite")
    tolerance = 16.0 * np.finfo(float).eps
    if (
        np.any(first < -1.0 - tolerance)
        or np.any(first > 1.0 + tolerance)
        or np.any(second < -1.0 - tolerance)
        or np.any(second > 1.0 + tolerance)
    ):
        raise ValueError("Pauli expectations must lie in [-1, 1]")

    exact = float(np.mean((first - second) ** 2))
    if shots_per_batch is None:
        return CrossBatchEstimate(exact, exact)

    a_first = sample_pauli_expectations(first, shots_per_batch, rng=rng)
    b_first = sample_pauli_expectations(second, shots_per_batch, rng=rng)
    a_second = sample_pauli_expectations(first, shots_per_batch, rng=rng)
    b_second = sample_pauli_expectations(second, shots_per_batch, rng=rng)
    diff_first = a_first - b_first
    diff_second = a_second - b_second
    signal = float(np.mean(diff_first * diff_second))
    raw = float(0.5 * (np.mean(diff_first**2) + np.mean(diff_second**2)))
    return CrossBatchEstimate(signal, raw)


def classify_blind(
    scores: pd.DataFrame,
    rule: FingerprintRule,
    n_boot: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Classify blinded recurrence-fingerprint score trajectories.

    The result contains exactly one row per ``blind_id``, ``shots_per_batch``,
    and ``measurement_seed``.  Each prediction is scoped to one ``physics_id``;
    complete carry/reset trajectories are paired by ``history_id`` and resampled
    as blocks, preserving every registered suffix and both conditions.

    ``exact_signal_mse`` may be present for later auditing, but this function
    neither reads nor validates it.  Known architecture-label and declared-
    window columns are rejected before any inference is performed.
    """

    if not isinstance(rule, FingerprintRule):
        raise TypeError("rule must be a FingerprintRule")
    if isinstance(n_boot, (bool, np.bool_)) or int(n_boot) != n_boot:
        raise ValueError("n_boot must be an integer >= 1")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be an integer >= 1")
    if (
        isinstance(bootstrap_seed, (bool, np.bool_))
        or int(bootstrap_seed) != bootstrap_seed
        or int(bootstrap_seed) < 0
    ):
        raise ValueError("bootstrap_seed must be a non-negative integer")

    clean = _validate_scores(scores, rule)
    grouped = list(clean.groupby(list(GROUP_COLUMNS), sort=False, dropna=False))
    grouped.sort(key=lambda item: _canonical_key(item[0]))

    rows: list[dict[str, Any]] = []
    for group_key, group in grouped:
        physics_ids = group["physics_id"].drop_duplicates().tolist()
        if len(physics_ids) != 1:
            raise ValueError(
                "each blind_id/shot/measurement group must belong to exactly "
                "one physics_id"
            )
        suffixes, physics_curves = _paired_curves(group, rule)
        point_curve = np.mean(
            [trajectories.mean(axis=0) for trajectories in physics_curves],
            axis=0,
        )
        # Case pseudonyms depend on the frozen config hash and therefore must
        # not perturb scientific uncertainty estimates.  Sharing this domain
        # key across case variants also keeps history resamples paired within a
        # physics block, shot tier, and measurement realization.
        bootstrap_key = (physics_ids[0], group_key[1], group_key[2])
        rng = _group_rng(int(bootstrap_seed), bootstrap_key)
        boot_curves = _hierarchical_bootstrap(
            physics_curves,
            n_boot=int(n_boot),
            rng=rng,
        )
        point = _curve_statistics(point_curve[None, ...], suffixes, rule)
        boot = _curve_statistics(boot_curves, suffixes, rule)
        rows.append(
            _classify_group(
                group_key=group_key,
                physics_id=physics_ids[0],
                suffixes=suffixes,
                physics_curves=physics_curves,
                point=point,
                boot=boot,
                rule=rule,
                n_boot=int(n_boot),
            )
        )

    result = pd.DataFrame(rows)
    if result.empty:
        # _validate_scores rejects empty input.  This is a defensive invariant.
        raise RuntimeError("no fingerprint groups were classified")
    result["inferred_cutoff"] = pd.array(result["inferred_cutoff"], dtype="Int64")
    return result.reset_index(drop=True)


def _validate_scores(scores: pd.DataFrame, rule: FingerprintRule) -> pd.DataFrame:
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a pandas.DataFrame")
    if scores.empty:
        raise ValueError("scores must be non-empty")
    if scores.columns.duplicated().any():
        raise ValueError("scores must not contain duplicate column names")

    normalized = {str(column).strip().casefold(): column for column in scores.columns}
    forbidden = sorted(set(normalized).intersection(FORBIDDEN_LABEL_COLUMNS))
    if forbidden:
        raise ValueError(f"forbidden label columns: {', '.join(forbidden)}")
    missing = sorted(REQUIRED_COLUMNS.difference(scores.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    identifiers = [*GROUP_COLUMNS, *TRAJECTORY_COLUMNS, "condition"]
    if scores[identifiers].isna().any(axis=None):
        raise ValueError("identifier and condition columns must not contain nulls")

    invalid_shot_labels = [
        value
        for value in scores["shots_per_batch"].drop_duplicates().tolist()
        if not _valid_shot_label(value)
    ]
    if invalid_shot_labels:
        raise ValueError(
            "shots_per_batch values must be 'infinity' or canonical positive "
            "integer strings"
        )

    clean = scores.loc[:, list(REQUIRED_COLUMNS)].copy()
    try:
        suffixes = pd.to_numeric(clean["suffix_length"], errors="raise").to_numpy(
            dtype=float
        )
        signals = pd.to_numeric(
            clean["cross_batch_signal_mse"], errors="raise"
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("suffix lengths and signal estimates must be numeric") from exc
    if not np.all(np.isfinite(suffixes)):
        raise ValueError("suffix_length values must be finite")
    if np.any(suffixes < 0) or np.any(suffixes != np.floor(suffixes)):
        raise ValueError("suffix_length values must be non-negative integers")
    if not np.all(np.isfinite(signals)):
        raise ValueError("cross_batch_signal_mse values must be finite")
    clean["suffix_length"] = suffixes.astype(int)
    clean["cross_batch_signal_mse"] = signals

    uniqueness_key = [
        *GROUP_COLUMNS,
        *TRAJECTORY_COLUMNS,
        "condition",
        "suffix_length",
    ]
    if clean.duplicated(uniqueness_key).any():
        raise ValueError("duplicate rows for a paired fingerprint trajectory")

    expected_conditions = {rule.carry_condition, rule.reset_condition}
    for group_key, group in clean.groupby(
        list(GROUP_COLUMNS), sort=False, dropna=False
    ):
        conditions = set(group["condition"].unique())
        if conditions != expected_conditions:
            raise ValueError(
                f"fingerprint group {group_key!r} must contain exactly the "
                "configured carry and reset conditions"
            )
    return clean


def _valid_shot_label(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value == INFINITY:
        return True
    return value.isascii() and value.isdigit() and str(int(value)) == value and int(value) >= 1


def _paired_curves(
    group: pd.DataFrame,
    rule: FingerprintRule,
) -> tuple[np.ndarray, list[np.ndarray]]:
    physics_curves: list[np.ndarray] = []
    expected_suffixes: tuple[int, ...] | None = None
    condition_order = (rule.carry_condition, rule.reset_condition)

    physics_groups = list(group.groupby("physics_id", sort=False, dropna=False))
    physics_groups.sort(key=lambda item: _canonical_key((item[0],)))
    for _, physics_group in physics_groups:
        history_curves: list[np.ndarray] = []
        history_groups = list(
            physics_group.groupby("history_id", sort=False, dropna=False)
        )
        history_groups.sort(key=lambda item: _canonical_key((item[0],)))
        for _, trajectory in history_groups:
            condition_suffixes: list[tuple[int, ...]] = []
            condition_values: list[np.ndarray] = []
            for condition in condition_order:
                selected = trajectory[trajectory["condition"] == condition].sort_values(
                    "suffix_length", kind="mergesort"
                )
                suffix_tuple = tuple(selected["suffix_length"].to_numpy(dtype=int))
                condition_suffixes.append(suffix_tuple)
                condition_values.append(
                    selected["cross_batch_signal_mse"].to_numpy(dtype=float)
                )
            if (
                not condition_suffixes[0]
                or condition_suffixes[0] != condition_suffixes[1]
            ):
                raise ValueError("incomplete paired fingerprint trajectory")
            if expected_suffixes is None:
                expected_suffixes = condition_suffixes[0]
            elif condition_suffixes[0] != expected_suffixes:
                raise ValueError("all paired trajectories must share one suffix grid")
            history_curves.append(np.stack(condition_values, axis=0))
        if not history_curves:
            raise ValueError("each physics unit must contain at least one history")
        physics_curves.append(np.stack(history_curves, axis=0))

    if expected_suffixes is None or not physics_curves:
        raise ValueError("each fingerprint group must contain paired trajectories")
    suffixes = np.asarray(expected_suffixes, dtype=int)
    required_cutoffs = set(range(rule.max_window + 1))
    if not required_cutoffs.issubset(set(suffixes.tolist())):
        raise ValueError(
            "suffix grid must contain every integer from 0 through max_window"
        )
    missing_tail = sorted(set(rule.tail_suffixes).difference(suffixes.tolist()))
    if missing_tail:
        raise ValueError(f"suffix grid is missing registered tail points: {missing_tail}")
    return suffixes, physics_curves


def _hierarchical_bootstrap(
    physics_curves: list[np.ndarray],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_physics = len(physics_curves)
    output = np.empty((n_boot, *physics_curves[0].shape[1:]), dtype=float)
    for bootstrap_index in range(n_boot):
        physics_draw = rng.integers(0, n_physics, size=n_physics)
        system_means: list[np.ndarray] = []
        for physics_index in physics_draw:
            trajectories = physics_curves[int(physics_index)]
            history_draw = rng.integers(
                0, len(trajectories), size=len(trajectories)
            )
            system_means.append(trajectories[history_draw].mean(axis=0))
        output[bootstrap_index] = np.mean(system_means, axis=0)
    return output


def _curve_statistics(
    curves: np.ndarray,
    suffixes: np.ndarray,
    rule: FingerprintRule,
) -> dict[str, np.ndarray]:
    carry = curves[:, 0, :]
    reset = curves[:, 1, :]
    zero_index = int(np.flatnonzero(suffixes == 0)[0])
    baseline = carry[:, zero_index]
    valid = np.abs(baseline) > np.finfo(float).tiny

    carry_ratio = np.full_like(carry, np.nan, dtype=float)
    reset_ratio = np.full_like(reset, np.nan, dtype=float)
    carry_ratio[valid] = carry[valid] / baseline[valid, None]
    reset_ratio[valid] = reset[valid] / baseline[valid, None]

    tail_mask = np.isin(suffixes, np.asarray(rule.tail_suffixes, dtype=int))
    carry_tail = carry_ratio[:, tail_mask]
    reset_tail = reset_ratio[:, tail_mask]
    carry_tail_mean = np.full(len(curves), np.nan, dtype=float)
    reset_tail_abs_mean = np.full(len(curves), np.nan, dtype=float)
    reset_tail_max = np.full(len(curves), np.nan, dtype=float)
    reset_contrast = np.full(len(curves), np.nan, dtype=float)
    carry_tail_mean[valid] = np.mean(carry_tail[valid], axis=1)
    reset_tail_abs_mean[valid] = np.mean(np.abs(reset_tail[valid]), axis=1)
    reset_tail_max[valid] = np.max(np.abs(reset_tail[valid]), axis=1)
    reset_contrast[valid] = (
        carry_tail_mean[valid] - reset_tail_abs_mean[valid]
    )

    statistics: dict[str, np.ndarray] = {
        "baseline_signal_mse": baseline,
        "recurrent_tail_mean_ratio": carry_tail_mean,
        "reset_tail_mean_abs_ratio": reset_tail_abs_mean,
        "reset_tail_max_abs_ratio": reset_tail_max,
        "reset_contrast_ratio": reset_contrast,
    }
    for cutoff in range(1, rule.max_window + 1):
        liveness_index = int(np.flatnonzero(suffixes == cutoff - 1)[0])
        cutoff_mask = suffixes >= cutoff
        cutoff_max = np.full(len(curves), np.nan, dtype=float)
        cutoff_max[valid] = np.max(
            np.abs(carry_ratio[valid][:, cutoff_mask]), axis=1
        )
        statistics[f"cutoff_{cutoff}_liveness"] = carry_ratio[:, liveness_index]
        statistics[f"cutoff_{cutoff}_equivalence"] = cutoff_max
    return statistics


def _classify_group(
    *,
    group_key: tuple[Any, ...],
    physics_id: Any,
    suffixes: np.ndarray,
    physics_curves: list[np.ndarray],
    point: dict[str, np.ndarray],
    boot: dict[str, np.ndarray],
    rule: FingerprintRule,
    n_boot: int,
) -> dict[str, Any]:
    metric_names = (
        "baseline_signal_mse",
        "recurrent_tail_mean_ratio",
        "reset_tail_mean_abs_ratio",
        "reset_tail_max_abs_ratio",
        "reset_contrast_ratio",
    )
    metrics: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    for name in metric_names:
        metrics[name] = float(point[name][0])
        intervals[name] = _confidence_interval(boot[name], rule.confidence)

    baseline_low, _ = intervals["baseline_signal_mse"]
    live = bool(baseline_low > rule.baseline_floor)
    low_signal = not live

    inferred_cutoff: int | None = None
    cutoff_liveness = float("nan")
    cutoff_liveness_low = float("nan")
    cutoff_liveness_high = float("nan")
    cutoff_equivalence = float("nan")
    cutoff_equivalence_low = float("nan")
    cutoff_equivalence_high = float("nan")
    qualifying_cutoffs: list[int] = []
    if live:
        for cutoff in range(1, rule.max_window + 1):
            live_name = f"cutoff_{cutoff}_liveness"
            equivalent_name = f"cutoff_{cutoff}_equivalence"
            live_ci = _confidence_interval(boot[live_name], rule.confidence)
            equivalent_ci = _confidence_interval(
                boot[equivalent_name], rule.confidence
            )
            if (
                live_ci[0] > rule.liveness_ratio
                and equivalent_ci[1] < rule.equivalence_ratio
            ):
                qualifying_cutoffs.append(cutoff)
        if qualifying_cutoffs:
            inferred_cutoff = min(qualifying_cutoffs)
            live_name = f"cutoff_{inferred_cutoff}_liveness"
            equivalent_name = f"cutoff_{inferred_cutoff}_equivalence"
            cutoff_liveness = float(point[live_name][0])
            cutoff_liveness_low, cutoff_liveness_high = _confidence_interval(
                boot[live_name], rule.confidence
            )
            cutoff_equivalence = float(point[equivalent_name][0])
            cutoff_equivalence_low, cutoff_equivalence_high = _confidence_interval(
                boot[equivalent_name], rule.confidence
            )

    carry_tail_present = bool(
        live
        and intervals["recurrent_tail_mean_ratio"][0]
        > rule.recurrent_tail_ratio
    )
    recurrent_qualifies = bool(
        carry_tail_present
        and intervals["reset_tail_max_abs_ratio"][1] < rule.reset_max_ratio
        and intervals["reset_contrast_ratio"][0] > rule.reset_contrast_ratio
    )
    reset_tail_persists = bool(
        carry_tail_present
        and intervals["reset_tail_mean_abs_ratio"][0]
        > rule.recurrent_tail_ratio
    )
    cutoff_compatible = inferred_cutoff is not None

    if low_signal:
        classification = FingerprintLabel.INDETERMINATE_LOW_SIGNAL.value
    elif recurrent_qualifies and not cutoff_compatible:
        classification = FingerprintLabel.RECURRENT_RESET_SENSITIVE.value
    elif cutoff_compatible and not recurrent_qualifies:
        classification = (
            f"{FingerprintLabel.FINITE_WINDOW_COMPATIBLE.value}_d{inferred_cutoff}"
        )
    elif not cutoff_compatible and carry_tail_present and reset_tail_persists:
        classification = FingerprintLabel.RECURRENT_RESET_INSENSITIVE.value
    else:
        classification = FingerprintLabel.INDETERMINATE.value

    row: dict[str, Any] = dict(zip(GROUP_COLUMNS, group_key, strict=True))
    row.update(
        {
            "physics_id": physics_id,
            "classification": classification,
            "inferred_cutoff": inferred_cutoff,
            "max_window": int(rule.max_window),
            "n_physics_units": int(len(physics_curves)),
            "n_trajectories": int(sum(len(item) for item in physics_curves)),
            "n_suffix_points": int(len(suffixes)),
            "n_tail_points": int(len(rule.tail_suffixes)),
            "n_boot": int(n_boot),
            "confidence": float(rule.confidence),
            "baseline_live": live,
            "low_signal": low_signal,
            "cutoff_compatible": cutoff_compatible,
            "n_qualifying_cutoffs": int(len(qualifying_cutoffs)),
            "sustained_tail": carry_tail_present,
            "stable_reset_contrast": recurrent_qualifies,
            "reset_tail_persists": reset_tail_persists,
            "inferred_cutoff_liveness_ratio": cutoff_liveness,
            "inferred_cutoff_liveness_ratio_ci_low": cutoff_liveness_low,
            "inferred_cutoff_liveness_ratio_ci_high": cutoff_liveness_high,
            "inferred_cutoff_max_abs_ratio": cutoff_equivalence,
            "inferred_cutoff_max_abs_ratio_ci_low": cutoff_equivalence_low,
            "inferred_cutoff_max_abs_ratio_ci_high": cutoff_equivalence_high,
        }
    )
    for name in metric_names:
        low, high = intervals[name]
        row[name] = metrics[name]
        row[f"{name}_ci_low"] = low
        row[f"{name}_ci_high"] = high
    return row


def _confidence_interval(
    values: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _canonical_key(values: tuple[Any, ...]) -> bytes:
    payload = [
        {"type": type(value).__qualname__, "value": str(value)} for value in values
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _group_rng(seed: int, group_key: tuple[Any, ...]) -> np.random.Generator:
    digest = hashlib.sha256(_canonical_key(group_key)).digest()
    words = [
        int.from_bytes(digest[offset : offset + 4], "little")
        for offset in range(0, len(digest), 4)
    ]
    seed_words = [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF]
    return np.random.default_rng(np.random.SeedSequence([*seed_words, *words]))
