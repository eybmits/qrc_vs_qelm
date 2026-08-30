"""Quantum-information bridge for finite-window QRC emulation.

For a fixed recent suffix, old histories induce an ensemble of recurrent-QRC
states.  This module connects two quantities for that ensemble:

* the conditional Holevo information retained by the quantum state; and
* the minimum squared risk of any predictor that receives only the suffix and
  is judged on a declared family of bounded observables.

Natural logarithms are used throughout.  If every observable has operator norm
at most one, quantum Pinsker gives ``history_risk <= 2 * holevo_nats``.  Thus a
positive observable history floor is a tomography-free certificate of retained
old-history information, while a vanishing Holevo quantity removes the history
floor for every such observable panel.

The module also implements a computational-basis dephasing intervention at the
prefix/suffix boundary.  Its exact population--coherence--interference
decomposition diagnoses which quantum-state sector carries the observable
history.  It is a mechanism diagnostic, not a quantum-advantage witness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from qrc_vs_qelm.core import liouvillian, readout


@dataclass(frozen=True)
class HolevoRiskBridge:
    """Exact diagnostics for one suffix-conditioned state ensemble."""

    n_states: int
    n_observables: int
    holevo_nats: float
    history_risk: float
    pinsker_upper_bound: float
    pinsker_slack: float
    certified_holevo_lower_bound: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class CoherenceSectorBridge:
    """Exact boundary-dephasing decomposition for one suffix ensemble."""

    history_risk: float
    population_history_risk: float
    coherence_history_risk: float
    population_coherence_cross_term: float
    dephasing_intervention_mse: float
    decomposition_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _probabilities(n_states: int, probabilities: np.ndarray | None) -> np.ndarray:
    if n_states < 1:
        raise ValueError("at least one state is required")
    if probabilities is None:
        return np.full(n_states, 1.0 / n_states, dtype=float)
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (n_states,):
        raise ValueError("probabilities must contain one value per state")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("probabilities must have positive sum")
    values = values / total
    return values


def _density_matrix(state: np.ndarray, *, atol: float = 2e-9) -> np.ndarray:
    matrix = np.asarray(state, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("state must be a square matrix")
    hermitian = 0.5 * (matrix + matrix.conj().T)
    if float(np.max(np.abs(matrix - hermitian))) > atol:
        raise ValueError("state is not Hermitian")
    trace = complex(np.trace(hermitian))
    if abs(trace.imag) > atol or abs(trace.real - 1.0) > atol:
        raise ValueError("state must have unit trace")
    eigenvalues = np.linalg.eigvalsh(hermitian)
    if float(np.min(eigenvalues)) < -atol:
        raise ValueError("state is not positive semidefinite")
    return hermitian


def von_neumann_entropy(state: np.ndarray, *, atol: float = 2e-9) -> float:
    """Return ``-Tr(rho log rho)`` in nats with numerical PSD validation."""

    matrix = _density_matrix(state, atol=atol)
    eigenvalues = np.clip(np.linalg.eigvalsh(matrix), 0.0, 1.0)
    positive = eigenvalues[eigenvalues > atol]
    return float(-np.sum(positive * np.log(positive)))


def conditional_holevo_nats(
    states: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> float:
    """Return the Holevo information of a suffix-conditioned state ensemble."""

    array = np.asarray(states, dtype=complex)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError("states must have shape (n_states, dim, dim)")
    checked = np.stack([_density_matrix(state) for state in array])
    weights = _probabilities(len(checked), probabilities)
    mean_state = np.einsum("i,ijk->jk", weights, checked)
    value = von_neumann_entropy(mean_state) - float(
        np.dot(weights, [von_neumann_entropy(state) for state in checked])
    )
    # Concavity makes this nonnegative; clip only roundoff-scale negatives.
    if value < -2e-9:
        raise RuntimeError("Holevo information became negative beyond tolerance")
    return float(max(0.0, value))


def _observable_matrices(
    observables: Sequence[readout.Observable | np.ndarray],
) -> np.ndarray:
    if not observables:
        raise ValueError("at least one observable is required")
    matrices = np.stack(
        [
            np.asarray(item.matrix if isinstance(item, readout.Observable) else item, complex)
            for item in observables
        ]
    )
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("observables must be square matrices of a common size")
    for matrix in matrices:
        if not np.allclose(matrix, matrix.conj().T, atol=2e-9, rtol=0):
            raise ValueError("observables must be Hermitian")
        if float(np.linalg.norm(matrix, ord=2)) > 1.0 + 2e-9:
            raise ValueError("observable operator norm exceeds one")
    return matrices


def _expectations(states: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    values = np.einsum("bij,aji->ba", states, matrices)
    if float(np.max(np.abs(values.imag))) > 2e-9:
        raise RuntimeError("observable expectations acquired a large imaginary part")
    return values.real


def observable_history_risk(
    states: np.ndarray,
    observables: Sequence[readout.Observable | np.ndarray],
    probabilities: np.ndarray | None = None,
) -> float:
    """Return Bayes MSE per observable for a fixed suffix ensemble."""

    array = np.asarray(states, dtype=complex)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError("states must have shape (n_states, dim, dim)")
    checked = np.stack([_density_matrix(state) for state in array])
    weights = _probabilities(len(checked), probabilities)
    matrices = _observable_matrices(observables)
    if matrices.shape[1:] != checked.shape[1:]:
        raise ValueError("state and observable dimensions do not match")
    values = _expectations(checked, matrices)
    mean = np.einsum("i,ij->j", weights, values)
    return float(np.einsum("i,ij,ij->", weights, values - mean, values - mean) / len(matrices))


def holevo_risk_bridge(
    states: np.ndarray,
    observables: Sequence[readout.Observable | np.ndarray],
    probabilities: np.ndarray | None = None,
    *,
    tolerance: float = 2e-9,
) -> HolevoRiskBridge:
    """Evaluate and audit ``H <= 2 chi`` for one conditional ensemble."""

    array = np.asarray(states, dtype=complex)
    matrices = _observable_matrices(observables)
    holevo = conditional_holevo_nats(array, probabilities)
    risk = observable_history_risk(array, list(matrices), probabilities)
    upper = 2.0 * holevo
    slack = upper - risk
    if slack < -tolerance:
        raise RuntimeError("observable history risk violated the Holevo--Pinsker bound")
    return HolevoRiskBridge(
        n_states=int(array.shape[0]),
        n_observables=int(matrices.shape[0]),
        holevo_nats=holevo,
        history_risk=risk,
        pinsker_upper_bound=upper,
        pinsker_slack=float(max(0.0, slack)),
        certified_holevo_lower_bound=0.5 * risk,
    )


def computational_dephasing(state: np.ndarray) -> np.ndarray:
    """Completely dephase a density matrix in the computational basis."""

    matrix = _density_matrix(state)
    return np.diag(np.diag(matrix)).astype(complex)


def apply_transfer(states: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    """Apply a column-vectorized superoperator to a state ensemble."""

    array = np.asarray(states, dtype=complex)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError("states must have shape (n_states, dim, dim)")
    dim = int(array.shape[1])
    operator = np.asarray(transfer, dtype=complex)
    if operator.shape != (dim * dim, dim * dim):
        raise ValueError("transfer shape does not match state dimension")
    evolved = np.stack(
        [liouvillian.unvec(operator @ liouvillian.vec(state), dim) for state in array]
    )
    return np.stack([_density_matrix(state, atol=2e-8) for state in evolved])


def coherence_sector_bridge(
    prefix_states: np.ndarray,
    transfer: np.ndarray,
    observables: Sequence[readout.Observable | np.ndarray],
    probabilities: np.ndarray | None = None,
    *,
    tolerance: float = 2e-8,
) -> CoherenceSectorBridge:
    """Decompose post-suffix observable history into population/coherence sectors.

    The boundary state is written exactly as ``rho = Delta(rho) +
    (I-Delta)(rho)``.  Linear suffix dynamics and readout give an exact
    population, coherence, and interference decomposition after centering over
    old histories.  ``dephasing_intervention_mse`` is the mean squared output
    change caused by applying ``Delta`` at the boundary.
    """

    prefix = np.asarray(prefix_states, dtype=complex)
    checked = np.stack([_density_matrix(state) for state in prefix])
    weights = _probabilities(len(checked), probabilities)
    matrices = _observable_matrices(observables)
    original_final = apply_transfer(checked, transfer)
    diagonal_prefix = np.stack([computational_dephasing(state) for state in checked])
    diagonal_final = apply_transfer(diagonal_prefix, transfer)
    y_total = _expectations(original_final, matrices)
    y_population = _expectations(diagonal_final, matrices)
    y_coherence = y_total - y_population

    def centered(values: np.ndarray) -> np.ndarray:
        return values - np.einsum("i,ij->j", weights, values)

    total_centered = centered(y_total)
    population_centered = centered(y_population)
    coherence_centered = centered(y_coherence)
    n_outputs = int(matrices.shape[0])

    def risk(values: np.ndarray) -> float:
        return float(np.einsum("i,ij,ij->", weights, values, values) / n_outputs)

    total_risk = risk(total_centered)
    population_risk = risk(population_centered)
    coherence_risk = risk(coherence_centered)
    cross = float(
        np.einsum(
            "i,ij,ij->", weights, population_centered, coherence_centered
        )
        / n_outputs
    )
    decomposition_error = abs(
        total_risk - (population_risk + coherence_risk + 2.0 * cross)
    )
    if decomposition_error > tolerance:
        raise RuntimeError("population--coherence decomposition failed")
    intervention = float(
        np.einsum("i,ij,ij->", weights, y_coherence, y_coherence) / n_outputs
    )
    return CoherenceSectorBridge(
        history_risk=total_risk,
        population_history_risk=population_risk,
        coherence_history_risk=coherence_risk,
        population_coherence_cross_term=cross,
        dephasing_intervention_mse=intervention,
        decomposition_error=decomposition_error,
    )
