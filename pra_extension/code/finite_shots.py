"""Operational finite-shot frontier for parallel-QELM readouts.

Single-site Pauli features are grouped into one global measurement setting per
snapshot and axis.  A shot produces all six single-qubit outcomes jointly, so
the feature covariance within a setting contains the exact same-axis two-site
correlations.  Settings are independent.

For a frozen linear readout with raw-feature weights ``W_q``, setting ``q``
contributes

    a_q / S_q,
    a_q = E[tr(W_q.T Cov_q W_q)] / n_outputs

to expected test MSE.  Ridge selection is shot aware: for every candidate
alpha we minimize noiseless validation MSE plus this exact expected validation
penalty.  After selection, the readout is refit on train plus validation and
the deployed allocation is recomputed from that fixed readout and the train-
plus-validation feature-state covariances.  Thus the target-aware rule is the
exact positive-integer minimizer of a disclosed design-set plug-in penalty,
not a test oracle.  Test inputs and targets never choose alpha or the deployed
allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from observable_memory import StressConfig, build_reservoir, default_config
from qrc_vs_qelm import canonical_emulation, emulation
from qrc_vs_qelm.core import readout, reservoirs
from qrc_vs_qelm.models import qelm


AXES = ("x", "y", "z")
MONTE_CARLO_NAMESPACE_CODE = 28_082_601
MONTE_CARLO_ALLOCATION_CODES = {"equal": 1, "optimal": 2}


@dataclass(frozen=True)
class ShotRoot:
    root_id: str
    teacher_coupling_seeds: Mapping[int, int]
    student_coupling_seed: int
    input_seed: int
    monte_carlo_seed: int


@dataclass(frozen=True)
class ShotStudyConfig:
    roots: tuple[ShotRoot, ...]
    teacher_n_qubits: tuple[int, ...]
    teacher_gamma: float
    window: int
    train_rows: int
    validation_rows: int
    test_rows: int
    burnin: int
    quantize_levels: int
    snapshot_budgets: tuple[int, ...]
    total_shots: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    monte_carlo_repeats: int
    student_n_qubits: int = 6
    max_snapshots: int = 4
    dt: float = 0.5
    h: float = 0.5
    h_in: float = 1.0
    coupling_scale: float = 1.0


@dataclass(frozen=True)
class FittedShotReadout:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray
    alpha: float
    validation_noiseless_mse: float
    validation_expected_penalty: float
    validation_objective: float
    selection_allocation: np.ndarray | None
    design_expected_penalty: float
    allocation: np.ndarray | None

    @property
    def raw_weights(self) -> np.ndarray:
        return self.weights / self.feature_scale[:, None]

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_mean) / self.feature_scale
        return standardized @ self.weights + self.intercept


def default_shot_config(profile: str = "full") -> ShotStudyConfig:
    """Return a deterministic canonical-like fast-fading shot study."""

    canonical_alphas = (
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
        return ShotStudyConfig(
            roots=(
                ShotRoot(
                    "shot_smoke01",
                    {2: 9_301_012},
                    student_coupling_seed=9_311_011,
                    input_seed=9_321_011,
                    monte_carlo_seed=9_331_011,
                ),
            ),
            teacher_n_qubits=(2,),
            teacher_gamma=4.0,
            window=2,
            train_rows=32,
            validation_rows=16,
            test_rows=24,
            burnin=12,
            quantize_levels=9,
            snapshot_budgets=(1, 2, 4),
            total_shots=(300, 1_200),
            alpha_grid=canonical_alphas,
            monte_carlo_repeats=3,
        )
    if profile == "n4":
        roots_n4 = tuple(
            ShotRoot(
                root_id=f"shotn4_{index:02d}",
                teacher_coupling_seeds={4: 9_500_000 + 100 * index + 4},
                student_coupling_seed=9_510_000 + 100 * index + 1,
                input_seed=9_520_000 + 100 * index + 1,
                monte_carlo_seed=9_530_000 + 100 * index + 1,
            )
            for index in range(1, 13)
        )
        return ShotStudyConfig(
            roots=roots_n4,
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
            alpha_grid=canonical_alphas,
            monte_carlo_repeats=16,
        )
    if profile != "full":
        raise ValueError("profile must be 'smoke', 'full', or 'n4'")
    roots = tuple(
        ShotRoot(
            root_id=f"shot{index:02d}",
            teacher_coupling_seeds={
                2: 9_300_000 + 100 * index + 2,
                3: 9_300_000 + 100 * index + 3,
            },
            student_coupling_seed=9_310_000 + 100 * index + 1,
            input_seed=9_320_000 + 100 * index + 1,
            monte_carlo_seed=9_330_000 + 100 * index + 1,
        )
        # shot01--shot03 were plumbing/calibration roots.  Keep the reported
        # twelve-root stress test disjoint from that first exploratory pass.
        for index in range(4, 16)
    )
    return ShotStudyConfig(
        roots=roots,
        teacher_n_qubits=(2, 3),
        teacher_gamma=4.0,
        window=6,
        train_rows=128,
        validation_rows=48,
        test_rows=96,
        burnin=48,
        quantize_levels=33,
        snapshot_budgets=(1, 2, 4),
        total_shots=(300, 1_200, 4_800, 19_200),
        alpha_grid=canonical_alphas,
        monte_carlo_repeats=16,
    )


def validate_shot_config(config: ShotStudyConfig) -> None:
    if not config.roots or not config.teacher_n_qubits:
        raise ValueError("roots and teacher sizes must be nonempty")
    root_ids = [root.root_id for root in config.roots]
    if len(set(root_ids)) != len(root_ids) or any(not value for value in root_ids):
        raise ValueError("root ids must be nonempty and unique")
    if config.student_n_qubits < config.window:
        raise ValueError("parallel encoding requires student_n_qubits >= window")
    if tuple(sorted(set(config.snapshot_budgets))) != config.snapshot_budgets:
        raise ValueError("snapshot budgets must be strictly increasing")
    if max(config.snapshot_budgets) != config.max_snapshots:
        raise ValueError("largest snapshot budget must equal max_snapshots")
    if any(budget < 1 for budget in config.snapshot_budgets):
        raise ValueError("snapshot budgets must be positive")
    if any(shots < 3 * max(config.snapshot_budgets) for shots in config.total_shots):
        raise ValueError("every setting must receive at least one shot")
    if any(rows < 1 for rows in (config.train_rows, config.validation_rows, config.test_rows)):
        raise ValueError("split sizes must be positive")
    if config.burnin < 1 or config.quantize_levels < 2:
        raise ValueError("burnin and quantize_levels are too small")
    if not config.alpha_grid or any(alpha < 0 or np.isnan(alpha) for alpha in config.alpha_grid):
        raise ValueError("invalid alpha grid")
    explicit_seeds: dict[int, tuple[str, str]] = {}
    for root in config.roots:
        if set(root.teacher_coupling_seeds) != set(config.teacher_n_qubits):
            raise ValueError(f"teacher seed coverage mismatch for {root.root_id}")
        named_seeds = {
            **{
                f"teacher_coupling_seed_N{int(n_qubits)}": int(seed)
                for n_qubits, seed in root.teacher_coupling_seeds.items()
            },
            "student_coupling_seed": int(root.student_coupling_seed),
            "input_seed": int(root.input_seed),
            "monte_carlo_seed": int(root.monte_carlo_seed),
        }
        for name, seed in named_seeds.items():
            if seed < 0:
                raise ValueError(f"seed {name} must be nonnegative in {root.root_id}")
            if seed in explicit_seeds:
                previous_root, previous_name = explicit_seeds[seed]
                raise ValueError(
                    "explicit seeds must be globally unique; "
                    f"{root.root_id}.{name} duplicates "
                    f"{previous_root}.{previous_name}"
                )
            explicit_seeds[seed] = (root.root_id, name)
    monte_carlo_seed_manifest(config)


def monte_carlo_entropy_tuple(
    *,
    root: ShotRoot,
    teacher_n_qubits: int,
    snapshot_budget: int,
    total_shots: int,
    allocation_mode: str,
) -> tuple[int, ...]:
    """Return the explicit entropy tuple for one Monte Carlo cell.

    Passing the full tuple to :class:`numpy.random.SeedSequence` avoids the
    collisions that arithmetic offsets can create across roots and resource
    cells.  The tuple itself, rather than a Python hash, is the auditable stream
    identity.
    """

    if allocation_mode not in MONTE_CARLO_ALLOCATION_CODES:
        raise ValueError("Monte Carlo streams require equal or optimal allocation")
    values = (
        int(root.monte_carlo_seed),
        int(MONTE_CARLO_NAMESPACE_CODE),
        int(teacher_n_qubits),
        int(snapshot_budget),
        int(total_shots),
        int(MONTE_CARLO_ALLOCATION_CODES[allocation_mode]),
    )
    if any(value < 0 for value in values):
        raise ValueError("Monte Carlo entropy coordinates must be nonnegative")
    return values


def monte_carlo_seed_manifest(
    config: ShotStudyConfig,
) -> tuple[dict[str, Any], ...]:
    """Build and validate the complete deterministic Monte Carlo stream manifest."""

    manifest: list[dict[str, Any]] = []
    seen_entropy: dict[tuple[int, ...], tuple[str, int, int, int, str]] = {}
    seen_states: dict[tuple[int, ...], tuple[str, int, int, int, str]] = {}
    for root, teacher_n, budget, total_shots, allocation_mode in product(
        config.roots,
        config.teacher_n_qubits,
        config.snapshot_budgets,
        config.total_shots,
        MONTE_CARLO_ALLOCATION_CODES,
    ):
        identity = (
            str(root.root_id),
            int(teacher_n),
            int(budget),
            int(total_shots),
            str(allocation_mode),
        )
        entropy = monte_carlo_entropy_tuple(
            root=root,
            teacher_n_qubits=int(teacher_n),
            snapshot_budget=int(budget),
            total_shots=int(total_shots),
            allocation_mode=str(allocation_mode),
        )
        if entropy in seen_entropy:
            raise ValueError(
                f"duplicate Monte Carlo entropy tuple for {identity} and "
                f"{seen_entropy[entropy]}"
            )
        seen_entropy[entropy] = identity
        state = tuple(
            int(value)
            for value in np.random.SeedSequence(entropy).generate_state(
                4, dtype=np.uint32
            )
        )
        if state in seen_states:
            raise ValueError(
                f"duplicate Monte Carlo SeedSequence state for {identity} and "
                f"{seen_states[state]}"
            )
        seen_states[state] = identity
        manifest.append(
            {
                "root_id": identity[0],
                "teacher_n_qubits": identity[1],
                "snapshot_budget": identity[2],
                "total_shots": identity[3],
                "allocation_mode": identity[4],
                "entropy": entropy,
                "state_uint32": state,
            }
        )
    return tuple(manifest)


def qelm_statevectors(
    channel: qelm.UnitaryQELMChannel,
    windows: np.ndarray,
) -> np.ndarray:
    """Return ``(rows, snapshots, amplitudes)`` for independent delay rows."""

    values = np.asarray(windows, float)
    if values.ndim != 2:
        raise ValueError("windows must be two dimensional")
    states = np.empty(
        (len(values), int(channel.n_snapshots), 2 ** int(channel.n_qubits)),
        dtype=complex,
    )
    unitaries = channel.unitaries
    for row, window in enumerate(values):
        initial = qelm.encode_window_statevector(
            window, int(channel.n_qubits), scheme="one_to_one"
        )
        for snapshot, unitary in enumerate(unitaries):
            states[row, snapshot] = unitary @ initial
    return states


def exact_single_site_features(states: np.ndarray, n_qubits: int) -> np.ndarray:
    """Flatten snapshot-major, axis-major, qubit-major exact expectations."""

    observables = readout.pauli_observables(
        int(n_qubits), axes=AXES, max_weight=1
    )
    matrices = np.stack([item.matrix for item in observables])
    values = np.real(
        np.einsum("rsi,kij,rsj->rsk", states.conj(), matrices, states)
    )
    return values.reshape(len(states), -1)


def select_nested_measurement_data(
    *,
    states: np.ndarray,
    features: np.ndarray,
    covariances: np.ndarray,
    max_snapshots: int,
    selected_snapshots: int,
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the canonical nested snapshot set ending at ``T``.

    With four maximal snapshots, the active indices are ``[3]`` for one,
    ``[1, 3]`` for two, and ``[0, 1, 2, 3]`` for four.  The returned states,
    feature blocks, and covariance blocks have the same local snapshot order.
    """

    values = np.asarray(states)
    covariance_values = np.asarray(covariances)
    if values.ndim != 3 or values.shape[1] != int(max_snapshots):
        raise ValueError("states do not contain max_snapshots blocks")
    if covariance_values.ndim != 5 or covariance_values.shape[1] != int(
        max_snapshots
    ):
        raise ValueError("covariances do not contain max_snapshots blocks")
    n_observables = len(AXES) * int(n_qubits)
    snapshot_indices = canonical_emulation.nested_snapshot_columns(
        max_snapshots=int(max_snapshots),
        selected_snapshots=int(selected_snapshots),
        n_observables=1,
    )
    selected_features = canonical_emulation.select_nested_snapshots(
        features,
        max_snapshots=int(max_snapshots),
        selected_snapshots=int(selected_snapshots),
        n_observables=n_observables,
    )
    return (
        values[:, snapshot_indices],
        selected_features,
        covariance_values[:, snapshot_indices],
        snapshot_indices,
    )


def setting_covariances(states: np.ndarray, n_qubits: int) -> np.ndarray:
    """Exact one-shot covariance for every snapshot/global-axis setting."""

    rows, snapshots, amplitudes = states.shape
    if amplitudes != 2 ** int(n_qubits):
        raise ValueError("state width does not match n_qubits")
    output = np.empty((rows, snapshots, len(AXES), n_qubits, n_qubits), float)
    for axis_index, axis in enumerate(AXES):
        singles = readout.pauli_observables(
            int(n_qubits), axes=(axis,), max_weight=1
        )
        single_matrices = np.stack([item.matrix for item in singles])
        means = np.real(
            np.einsum("rsi,kij,rsj->rsk", states.conj(), single_matrices, states)
        )
        second = np.empty((rows, snapshots, n_qubits, n_qubits), float)
        for first in range(n_qubits):
            second[:, :, first, first] = 1.0
            for second_index in range(first + 1, n_qubits):
                pair = reservoirs.two_site_pauli(
                    axis, first, second_index, int(n_qubits)
                )
                expectation = np.real(
                    np.einsum("rsi,ij,rsj->rs", states.conj(), pair, states)
                )
                second[:, :, first, second_index] = expectation
                second[:, :, second_index, first] = expectation
        covariance = second - np.einsum("rsi,rsj->rsij", means, means)
        output[:, :, axis_index] = 0.5 * (
            covariance + np.swapaxes(covariance, -1, -2)
        )
    return output


def setting_contributions(
    covariances: np.ndarray,
    raw_weights: np.ndarray,
    *,
    snapshot_budget: int,
    n_qubits: int,
) -> np.ndarray:
    """Return the per-setting coefficients ``a_q`` for expected MSE penalty."""

    expected_width = int(snapshot_budget) * len(AXES) * int(n_qubits)
    if raw_weights.shape[0] != expected_width:
        raise ValueError("raw weight width does not match active settings")
    if covariances.shape[1] < int(snapshot_budget):
        raise ValueError("covariance tensor has too few snapshots")
    n_outputs = int(raw_weights.shape[1])
    contributions: list[float] = []
    for snapshot, axis_index in product(range(snapshot_budget), range(len(AXES))):
        start = (snapshot * len(AXES) + axis_index) * int(n_qubits)
        weights = raw_weights[start : start + int(n_qubits)]
        covariance = covariances[:, snapshot, axis_index]
        row_penalty = np.einsum(
            "io,rij,jo->r", weights, covariance, weights, optimize=True
        ) / float(n_outputs)
        value = float(np.mean(row_penalty))
        if value < -1e-12:
            raise RuntimeError("physical setting covariance produced negative penalty")
        contributions.append(max(value, 0.0))
    return np.asarray(contributions, float)


def equal_allocation(total_shots: int, n_settings: int) -> np.ndarray:
    """Deterministically split an integer budget as evenly as possible."""

    if int(total_shots) < int(n_settings) or int(n_settings) < 1:
        raise ValueError("total_shots must be at least n_settings")
    base, remainder = divmod(int(total_shots), int(n_settings))
    allocation = np.full(int(n_settings), base, dtype=int)
    allocation[:remainder] += 1
    return allocation


def optimal_allocation(contributions: np.ndarray, total_shots: int) -> np.ndarray:
    """Exactly minimize ``sum(a_q/S_q)`` over positive integer allocations."""

    values = np.asarray(contributions, float)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0):
        raise ValueError("contributions must be a nonnegative vector")
    if int(total_shots) < len(values):
        raise ValueError("total_shots must give every setting at least one shot")
    allocation = np.ones(len(values), dtype=int)
    for _ in range(int(total_shots) - len(values)):
        marginal = values / (allocation * (allocation + 1))
        allocation[int(np.argmax(marginal))] += 1
    return allocation


def expected_shot_penalty(
    contributions: np.ndarray,
    allocation: np.ndarray,
) -> float:
    values = np.asarray(contributions, float)
    shots = np.asarray(allocation, int)
    if values.shape != shots.shape or np.any(shots < 1):
        raise ValueError("contributions and positive allocation must align")
    return float(np.sum(values / shots))


def _fit_fixed_alpha(
    standardized_features: np.ndarray,
    targets: np.ndarray,
    rows: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    return emulation._fit_centered_multi(
        standardized_features[rows], targets[rows], float(alpha)
    )


def select_shot_aware_readout(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    covariances: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    alphas: Sequence[float],
    snapshot_budget: int,
    n_qubits: int,
    allocation_mode: str,
    total_shots: int | None,
) -> FittedShotReadout:
    """Select alpha on exact validation MSE plus expected validation shot MSE."""

    if allocation_mode not in {"noiseless", "equal", "optimal"}:
        raise ValueError("unknown allocation mode")
    if allocation_mode == "noiseless" and total_shots is not None:
        raise ValueError("noiseless mode takes no shot budget")
    if allocation_mode != "noiseless" and total_shots is None:
        raise ValueError("finite-shot modes require total_shots")
    x = np.asarray(features, float)
    y = np.asarray(targets, float)
    train = np.asarray(train_idx, int)
    validation = np.asarray(validation_idx, int)
    mean = np.mean(x[train], axis=0)
    scale = np.std(x[train], axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (x - mean) / scale
    best: tuple[
        tuple[float, float, int],
        float,
        float,
        np.ndarray | None,
        float,
    ] | None = None
    for alpha_index, alpha_value in enumerate(alphas):
        alpha = float(alpha_value)
        weights, intercept = _fit_fixed_alpha(standardized, y, train, alpha)
        prediction = standardized[validation] @ weights + intercept
        validation_mse = float(np.mean((y[validation] - prediction) ** 2))
        allocation = None
        penalty = 0.0
        if allocation_mode != "noiseless":
            contributions = setting_contributions(
                covariances[validation],
                weights / scale[:, None],
                snapshot_budget=int(snapshot_budget),
                n_qubits=int(n_qubits),
            )
            allocation = (
                equal_allocation(int(total_shots), len(contributions))
                if allocation_mode == "equal"
                else optimal_allocation(contributions, int(total_shots))
            )
            penalty = expected_shot_penalty(contributions, allocation)
        objective = validation_mse + penalty
        # Deterministic tie rule: lower exact objective, then lower noiseless
        # validation MSE, then the earlier alpha in the declared grid.
        candidate = (
            (objective, validation_mse, alpha_index),
            validation_mse,
            penalty,
            None if allocation is None else allocation.copy(),
            alpha,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    (
        selection_key,
        selected_validation_mse,
        selected_penalty,
        selected_allocation,
        alpha,
    ) = best
    train_validation = np.concatenate([train, validation])
    weights, intercept = _fit_fixed_alpha(
        standardized, y, train_validation, float(alpha)
    )
    deployed_allocation = None
    design_expected_penalty = 0.0
    if allocation_mode != "noiseless":
        design_contributions = setting_contributions(
            covariances[train_validation],
            weights / scale[:, None],
            snapshot_budget=int(snapshot_budget),
            n_qubits=int(n_qubits),
        )
        deployed_allocation = (
            equal_allocation(int(total_shots), len(design_contributions))
            if allocation_mode == "equal"
            else optimal_allocation(design_contributions, int(total_shots))
        )
        design_expected_penalty = expected_shot_penalty(
            design_contributions, deployed_allocation
        )
    return FittedShotReadout(
        feature_mean=mean,
        feature_scale=scale,
        weights=np.asarray(weights, float),
        intercept=np.asarray(intercept, float),
        alpha=float(alpha),
        validation_noiseless_mse=float(selected_validation_mse),
        validation_expected_penalty=float(selected_penalty),
        validation_objective=float(selection_key[0]),
        selection_allocation=(
            None
            if selected_allocation is None
            else np.asarray(selected_allocation, int)
        ),
        design_expected_penalty=float(design_expected_penalty),
        allocation=(
            None
            if deployed_allocation is None
            else np.asarray(deployed_allocation, int)
        ),
    )


def _measurement_rotation(axis: str) -> np.ndarray:
    root_two = np.sqrt(2.0)
    if axis == "z":
        return np.eye(2, dtype=complex)
    if axis == "x":
        return np.asarray([[1, 1], [1, -1]], complex) / root_two
    if axis == "y":
        return np.asarray([[1, -1j], [1, 1j]], complex) / root_two
    raise ValueError(f"unknown axis: {axis!r}")


def joint_setting_probabilities(
    state: np.ndarray,
    axis: str,
    n_qubits: int,
) -> np.ndarray:
    """Exact multinomial probabilities for one global Pauli-basis setting."""

    rotation = _measurement_rotation(axis)
    full = rotation
    for _ in range(1, int(n_qubits)):
        full = np.kron(full, rotation)
    amplitudes = full @ np.asarray(state, complex)
    probabilities = np.abs(amplitudes) ** 2
    probabilities = probabilities / np.sum(probabilities)
    return probabilities.real


def outcome_table(n_qubits: int) -> np.ndarray:
    indices = np.arange(2 ** int(n_qubits), dtype=int)
    bits = (
        indices[:, None]
        >> np.arange(int(n_qubits) - 1, -1, -1, dtype=int)[None, :]
    ) & 1
    return 1.0 - 2.0 * bits.astype(float)


def monte_carlo_frozen_penalty(
    *,
    states: np.ndarray,
    exact_features: np.ndarray,
    raw_weights: np.ndarray,
    snapshot_budget: int,
    allocation: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float]:
    """Validate expected penalty by joint multinomial test-feature sampling."""

    n_rows, _, width = states.shape
    n_qubits = int(round(np.log2(width)))
    outcomes = outcome_table(n_qubits)
    active_width = int(snapshot_budget) * len(AXES) * n_qubits
    if exact_features.shape != (n_rows, active_width):
        raise ValueError("exact feature shape mismatch")
    if len(allocation) != int(snapshot_budget) * len(AXES):
        raise ValueError("allocation shape mismatch")
    probabilities = np.empty(
        (n_rows, int(snapshot_budget), len(AXES), len(outcomes)), float
    )
    for row, snapshot, axis_index in product(
        range(n_rows), range(snapshot_budget), range(len(AXES))
    ):
        probabilities[row, snapshot, axis_index] = joint_setting_probabilities(
            states[row, snapshot], AXES[axis_index], n_qubits
        )
    penalties = np.empty(int(repeats), float)
    for repeat in range(int(repeats)):
        noisy = exact_features.copy()
        setting = 0
        for snapshot, axis_index in product(
            range(snapshot_budget), range(len(AXES))
        ):
            shots = int(allocation[setting])
            start = setting * n_qubits
            for row in range(n_rows):
                counts = rng.multinomial(
                    shots, probabilities[row, snapshot, axis_index]
                )
                noisy[row, start : start + n_qubits] = (
                    counts @ outcomes / float(shots)
                )
            setting += 1
        prediction_noise = (noisy - exact_features) @ raw_weights
        penalties[repeat] = float(np.mean(prediction_noise**2))
    return float(np.mean(penalties)), float(np.std(penalties, ddof=1)) if repeats > 1 else 0.0


def _teacher_and_student_data(
    config: ShotStudyConfig,
    root: ShotRoot,
    teacher_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_rows = config.train_rows + config.validation_rows + config.test_rows
    total_inputs = config.burnin + config.window - 1 + total_rows
    rng = np.random.default_rng(int(root.input_seed))
    inputs = rng.integers(
        0, int(config.quantize_levels), size=total_inputs
    ).astype(float) / (int(config.quantize_levels) - 1)
    teacher_config: StressConfig = default_config("smoke")
    teacher_config = StressConfig(
        roots=teacher_config.roots,
        n_qubits=(int(teacher_n),),
        dissipator_families=("local_loss",),
        gammas=(float(config.teacher_gamma),),
        windows=(int(config.window),),
        observable_panels=teacher_config.observable_panels,
        quantize_levels=int(config.quantize_levels),
        suffix_groups=min(4, int(config.quantize_levels)),
        prefix_branches=3,
        prefix_burnin=max(2, int(config.burnin)),
        dt=float(config.dt),
        h=float(config.h),
        h_in=float(config.h_in),
        coupling_scale=float(config.coupling_scale),
    )
    teacher, _ = build_reservoir(
        n_qubits=int(teacher_n),
        family="local_loss",
        gamma=float(config.teacher_gamma),
        coupling_seed=int(root.teacher_coupling_seeds[int(teacher_n)]),
        config=teacher_config,
    )
    target_observables = readout.pauli_observables(
        int(teacher_n), axes=AXES, max_weight=2
    )
    first = int(config.burnin + config.window - 1)
    targets = teacher.run(inputs, target_observables, washout=first, n_virtual=1)
    delays = qelm.delay_embed(inputs, int(config.window))[first:]

    student_rng = np.random.default_rng(int(root.student_coupling_seed))
    couplings = reservoirs.random_couplings(
        int(config.student_n_qubits), float(config.coupling_scale), student_rng
    )
    hamiltonian = reservoirs.ising_xx_hamiltonian(
        couplings, float(config.h), int(config.student_n_qubits)
    )
    channel = qelm.UnitaryQELMChannel.from_hamiltonian(
        n_qubits=int(config.student_n_qubits),
        H_static=hamiltonian,
        T_evolve=float(config.dt),
        n_snapshots=int(config.max_snapshots),
    )
    states = qelm_statevectors(channel, delays)
    features = exact_single_site_features(states, int(config.student_n_qubits))
    if len(targets) != total_rows or len(features) != total_rows:
        raise RuntimeError("split construction produced the wrong row count")
    return targets, states, features


def _alpha_fields(alpha: float) -> dict[str, Any]:
    return {
        "selected_alpha": "+inf" if np.isposinf(alpha) else format(alpha, ".12g"),
        "intercept_only_selected": bool(np.isposinf(alpha)),
    }


def run_shot_study(config: ShotStudyConfig) -> pd.DataFrame:
    """Run the shot-aware operational frontier with untouched test targets."""

    validate_shot_config(config)
    rows: list[dict[str, Any]] = []
    n = int(config.student_n_qubits)
    for root in config.roots:
        for teacher_n in config.teacher_n_qubits:
            targets, states, features_max = _teacher_and_student_data(
                config, root, int(teacher_n)
            )
            covariances = setting_covariances(states, n)
            train = np.arange(0, int(config.train_rows), dtype=int)
            validation = np.arange(
                int(config.train_rows),
                int(config.train_rows + config.validation_rows),
                dtype=int,
            )
            test = np.arange(
                int(config.train_rows + config.validation_rows),
                len(targets),
                dtype=int,
            )
            target_variance = float(
                np.mean((targets[test] - np.mean(targets[test], axis=0)) ** 2)
            )
            for budget in config.snapshot_budgets:
                (
                    active_states,
                    features,
                    active_covariances,
                    snapshot_indices,
                ) = select_nested_measurement_data(
                    states=states,
                    features=features_max,
                    covariances=covariances,
                    max_snapshots=int(config.max_snapshots),
                    selected_snapshots=int(budget),
                    n_qubits=n,
                )
                width = int(features.shape[1])
                conditions: list[tuple[str, int | None]] = [("noiseless", None)]
                conditions.extend(
                    (mode, int(total))
                    for total in config.total_shots
                    for mode in ("equal", "optimal")
                )
                for allocation_mode, total_shots in conditions:
                    model = select_shot_aware_readout(
                        features=features,
                        targets=targets,
                        covariances=active_covariances,
                        train_idx=train,
                        validation_idx=validation,
                        alphas=config.alpha_grid,
                        snapshot_budget=int(budget),
                        n_qubits=n,
                        allocation_mode=allocation_mode,
                        total_shots=total_shots,
                    )
                    prediction = model.predict(features[test])
                    noiseless_mse = float(np.mean((targets[test] - prediction) ** 2))
                    test_contributions = setting_contributions(
                        active_covariances[test],
                        model.raw_weights,
                        snapshot_budget=int(budget),
                        n_qubits=n,
                    )
                    shot_penalty = 0.0
                    expected_mse = noiseless_mse
                    mc_mean = mc_std = None
                    allocation_list: list[int] | None = None
                    selection_allocation_list: list[int] | None = None
                    if model.allocation is not None:
                        allocation_list = [int(value) for value in model.allocation]
                        assert model.selection_allocation is not None
                        selection_allocation_list = [
                            int(value) for value in model.selection_allocation
                        ]
                        shot_penalty = expected_shot_penalty(
                            test_contributions, model.allocation
                        )
                        expected_mse += shot_penalty
                        mc_entropy = monte_carlo_entropy_tuple(
                            root=root,
                            teacher_n_qubits=int(teacher_n),
                            snapshot_budget=int(budget),
                            total_shots=int(total_shots),
                            allocation_mode=allocation_mode,
                        )
                        mc_rng = np.random.default_rng(
                            np.random.SeedSequence(mc_entropy)
                        )
                        mc_mean, mc_std = monte_carlo_frozen_penalty(
                            states=active_states[test],
                            exact_features=features[test],
                            raw_weights=model.raw_weights,
                            snapshot_budget=int(budget),
                            allocation=model.allocation,
                            rng=mc_rng,
                            repeats=int(config.monte_carlo_repeats),
                        )
                    rows.append(
                        {
                            "root_id": root.root_id,
                            "teacher_n_qubits": int(teacher_n),
                            "teacher_gamma": float(config.teacher_gamma),
                            "student_n_qubits": n,
                            "window": int(config.window),
                            "snapshot_budget": int(budget),
                            "feature_budget": int(width),
                            "active_settings": int(3 * budget),
                            "snapshot_indices_zero_based": str(
                                [int(value) for value in snapshot_indices]
                            ),
                            "allocation_mode": allocation_mode,
                            "total_shots": total_shots,
                            "allocation_json": (
                                None if allocation_list is None else str(allocation_list)
                            ),
                            "validation_selection_allocation_json": (
                                None
                                if selection_allocation_list is None
                                else str(selection_allocation_list)
                            ),
                            "minimum_setting_shots": (
                                None if allocation_list is None else min(allocation_list)
                            ),
                            "maximum_setting_shots": (
                                None if allocation_list is None else max(allocation_list)
                            ),
                            **_alpha_fields(model.alpha),
                            "validation_noiseless_mse": model.validation_noiseless_mse,
                            "validation_expected_shot_penalty_mse": model.validation_expected_penalty,
                            "validation_expected_objective_mse": model.validation_objective,
                            "design_expected_shot_penalty_mse": model.design_expected_penalty,
                            "test_target_variance": target_variance,
                            "test_noiseless_mse": noiseless_mse,
                            "test_noiseless_nrmse": float(
                                np.sqrt(noiseless_mse / target_variance)
                            ),
                            "test_expected_shot_penalty_mse": shot_penalty,
                            "test_expected_total_mse": expected_mse,
                            "test_expected_total_nrmse": float(
                                np.sqrt(expected_mse / target_variance)
                            ),
                            "mc_frozen_penalty_mean_mse": mc_mean,
                            "mc_frozen_penalty_std_mse": mc_std,
                            "monte_carlo_entropy": (
                                None
                                if model.allocation is None
                                else str([int(value) for value in mc_entropy])
                            ),
                            "mc_repeats": (
                                0 if model.allocation is None else int(config.monte_carlo_repeats)
                            ),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        [
            "root_id",
            "teacher_n_qubits",
            "snapshot_budget",
            "allocation_mode",
            "total_shots",
        ],
        kind="mergesort",
        na_position="first",
    ).reset_index(drop=True)


def summarize_shot_results(results: pd.DataFrame) -> dict[str, Any]:
    """Summarize operational B-by-shot surfaces without pseudo-replication."""

    finite = results[results["allocation_mode"].ne("noiseless")].copy()
    keys = ["teacher_n_qubits", "snapshot_budget", "allocation_mode", "total_shots"]
    cells: list[dict[str, Any]] = []
    for key, group in finite.groupby(keys, sort=True):
        teacher_n_qubits, snapshot_budget, allocation_mode, total_shots = key
        cells.append(
            {
                "teacher_n_qubits": int(teacher_n_qubits),
                "snapshot_budget": int(snapshot_budget),
                "allocation_mode": str(allocation_mode),
                "total_shots": int(total_shots),
                "n_roots": int(group["root_id"].nunique()),
                "median_expected_test_nrmse": float(
                    np.median(group["test_expected_total_nrmse"])
                ),
                "median_noiseless_test_nrmse_under_selected_alpha": float(
                    np.median(group["test_noiseless_nrmse"])
                ),
                "median_expected_shot_penalty_mse": float(
                    np.median(group["test_expected_shot_penalty_mse"])
                ),
                "median_mc_to_exact_penalty_ratio": float(
                    np.median(
                        group["mc_frozen_penalty_mean_mse"]
                        / group["test_expected_shot_penalty_mse"]
                    )
                ),
                "intercept_only_roots": int(np.sum(group["intercept_only_selected"])),
            }
        )
    noiseless_cells: list[dict[str, Any]] = []
    noiseless = results[results["allocation_mode"].eq("noiseless")]
    for key, group in noiseless.groupby(
        ["teacher_n_qubits", "snapshot_budget"], sort=True
    ):
        noiseless_cells.append(
            {
                "teacher_n_qubits": int(key[0]),
                "snapshot_budget": int(key[1]),
                "n_roots": int(group["root_id"].nunique()),
                "median_test_nrmse": float(np.median(group["test_noiseless_nrmse"])),
                "selected_alpha_counts": {
                    str(alpha): int(count)
                    for alpha, count in group["selected_alpha"].value_counts().items()
                },
            }
        )
    return {
        "schema_version": 1,
        "study": "pra_finite_shot_qelm_extension",
        "status": "EXPLORATORY_STRESS_TEST",
        "measurement_model": (
            "joint multinomial single-site XYZ outcomes grouped by snapshot and axis"
        ),
        "selection_rule": (
            "alpha minimizes noiseless validation MSE plus exact expected validation "
            "shot penalty under the train-fit readout; after alpha selection, the "
            "readout is refit on train plus validation and the deployed allocation "
            "exactly minimizes its train-plus-validation plug-in integer shot penalty; "
            "test inputs and targets remain untouched"
        ),
        "counts": {
            "rows": int(len(results)),
            "roots": int(results["root_id"].nunique()),
            "teacher_sizes": sorted(
                int(value) for value in results["teacher_n_qubits"].unique()
            ),
        },
        "noiseless_cells": noiseless_cells,
        "finite_shot_cells": cells,
    }
