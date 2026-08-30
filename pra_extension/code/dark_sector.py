"""Protected--excited--visible memory in a two-qubit QRC.

The generic finite-window obstruction is elementary.  If two old histories
produce states with different expectation values of an observable ``P``, a
common suffix cannot erase that difference whenever every suffix channel obeys
``Phi_u^dagger(P) = P``.  A reset-on-window student receives the same suffix for
both histories, hence gives the same prediction.  For equally likely branches
its optimal squared error is at least ``(Delta <P>)**2 / 4`` and its worst-case
absolute error is at least ``abs(Delta <P>) / 2``.

This module supplies a genuinely quantum realization.  For two qubits under a
permutation-symmetric Hamiltonian and collective amplitude loss, the entangled
singlet projector

    P_- = |psi_-><psi_-| = (I - XX - YY - ZZ) / 4

is exactly conserved.  Symmetric driving from ``|00>`` cannot populate it, but
one addressable prefix pulse can.  The resulting old-history signal survives
an arbitrarily long symmetric suffix and is visible through two-qubit Pauli
correlators.  The construction is an emulation obstruction, not a claim of
quantum computational advantage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import expm

from qrc_vs_qelm.core import dissipators, liouvillian, operators, reservoirs


@dataclass(frozen=True)
class CommonSuffixBound:
    """Best suffix-only errors for a balanced two-history witness."""

    first_target: float
    second_target: float
    target_gap: float
    optimal_common_prediction: float
    minimum_mean_squared_error: float
    minimum_worst_case_absolute_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SingletAudit:
    """Algebraic residuals for the protected two-qubit mode."""

    pauli_decomposition_error: float
    swap_antisymmetry_error: float
    hamiltonian_commutator_error: float
    jump_annihilation_error: float
    adjoint_invariance_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def ket_density(vector: np.ndarray) -> np.ndarray:
    """Return the normalized pure-state density matrix of ``vector``."""

    ket = np.asarray(vector, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(ket))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("state vector must have positive finite norm")
    ket = ket / norm
    return np.outer(ket, ket.conj())


def zero_state() -> np.ndarray:
    """Return ``|00><00|`` in the repository's leftmost-qubit convention."""

    return ket_density(np.array([1.0, 0.0, 0.0, 0.0], dtype=complex))


def singlet_ket() -> np.ndarray:
    """Return ``(|01> - |10>) / sqrt(2)``."""

    return np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)


def singlet_projector() -> np.ndarray:
    """Return the entangled two-qubit singlet projector ``P_-``."""

    return ket_density(singlet_ket())


def swap_operator() -> np.ndarray:
    """Return the two-qubit swap operator."""

    value = np.zeros((4, 4), dtype=complex)
    for left in range(2):
        for right in range(2):
            source = 2 * left + right
            target = 2 * right + left
            value[target, source] = 1.0
    return value


def singlet_pauli_projector() -> np.ndarray:
    """Return ``(II - XX - YY - ZZ) / 4``."""

    ii = np.eye(4, dtype=complex)
    correlations = sum(
        operators.two_site_pauli(axis, 0, 1, 2) for axis in ("x", "y", "z")
    )
    return (ii - correlations) / 4.0


def expectation(state: np.ndarray, observable: np.ndarray) -> float:
    """Return a real expectation value with a Hermiticity audit."""

    rho = np.asarray(state, dtype=complex)
    matrix = np.asarray(observable, dtype=complex)
    if rho.shape != matrix.shape or rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("state and observable must be square matrices of equal size")
    value = complex(np.trace(matrix @ rho))
    if abs(value.imag) > 2e-9:
        raise RuntimeError("expectation value acquired a large imaginary part")
    return float(value.real)


def addressable_x_pulse(state: np.ndarray, angle: float, site: int = 0) -> np.ndarray:
    """Apply ``exp(-i angle X_site / 2)`` to a two-qubit state."""

    if site not in (0, 1):
        raise ValueError("the two-qubit pulse site must be 0 or 1")
    if not np.isfinite(angle):
        raise ValueError("pulse angle must be finite")
    generator = operators.pauli_op("x", site, 2)
    unitary = expm(-0.5j * float(angle) * generator)
    rho = np.asarray(state, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError("state must be a two-qubit density matrix")
    return unitary @ rho @ unitary.conj().T


def common_suffix_bound(first_target: float, second_target: float) -> CommonSuffixBound:
    """Return the exact balanced Bayes and worst-case suffix-only error floors."""

    values = np.asarray([first_target, second_target], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("targets must be finite")
    gap = float(values[1] - values[0])
    midpoint = float(np.mean(values))
    return CommonSuffixBound(
        first_target=float(values[0]),
        second_target=float(values[1]),
        target_gap=gap,
        optimal_common_prediction=midpoint,
        minimum_mean_squared_error=float(gap * gap / 4.0),
        minimum_worst_case_absolute_error=float(abs(gap) / 2.0),
    )


def partial_transpose(state: np.ndarray, subsystem: int = 1) -> np.ndarray:
    """Partially transpose one subsystem of a two-qubit state."""

    rho = np.asarray(state, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError("state must be a two-qubit density matrix")
    if subsystem not in (0, 1):
        raise ValueError("subsystem must be 0 or 1")
    tensor = rho.reshape(2, 2, 2, 2)
    if subsystem == 0:
        transposed = tensor.transpose(2, 1, 0, 3)
    else:
        transposed = tensor.transpose(0, 3, 2, 1)
    return transposed.reshape(4, 4)


def negativity(state: np.ndarray, subsystem: int = 1) -> float:
    """Return the two-qubit entanglement negativity."""

    eigenvalues = np.linalg.eigvalsh(partial_transpose(state, subsystem))
    value = float(np.sum(np.abs(eigenvalues[eigenvalues < 0.0])))
    return max(0.0, value)


def adjoint_action(superoperator: np.ndarray, observable: np.ndarray) -> np.ndarray:
    """Apply the Hilbert--Schmidt adjoint of a column-vectorized map."""

    matrix = np.asarray(observable, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("observable must be square")
    dim = int(matrix.shape[0])
    transfer = np.asarray(superoperator, dtype=complex)
    if transfer.shape != (dim * dim, dim * dim):
        raise ValueError("superoperator shape does not match observable")
    return liouvillian.unvec(transfer.conj().T @ liouvillian.vec(matrix), dim)


def lindblad_protection_defect_upper_bound(
    generator: np.ndarray,
    observable: np.ndarray,
    dt: float,
) -> float:
    """Bound one-step invariance loss by ``dt ||L^dagger(P)||_infty``.

    For a Lindblad semigroup, the adjoint channel is positive and unital, hence
    contractive in operator norm on Hermitian observables.  Integrating the
    Heisenberg equation therefore gives this conservative bound uniformly over
    the input step represented by ``generator``.
    """

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    image = adjoint_action(generator, observable)
    if np.linalg.norm(image - image.conj().T, ord="fro") > 1e-10:
        raise ValueError("adjoint generator image must be Hermitian")
    return float(dt * np.linalg.norm(image, ord=2))


def robust_common_suffix_gap_bound(
    initial_gap: float,
    protection_defect: float,
    suffix_length: int,
    state_trace_distance: float = 2.0,
) -> float:
    """Return the approximate-protection lower bound on a common-suffix gap."""

    values = np.asarray(
        [initial_gap, protection_defect, state_trace_distance], dtype=float
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("gap, defect, and trace distance must be finite")
    if protection_defect < 0 or state_trace_distance < 0 or state_trace_distance > 2:
        raise ValueError("defect must be nonnegative and trace distance must lie in [0,2]")
    if int(suffix_length) != suffix_length or suffix_length < 0:
        raise ValueError("suffix length must be a nonnegative integer")
    return float(
        max(
            0.0,
            abs(initial_gap)
            - int(suffix_length) * protection_defect * state_trace_distance,
        )
    )


def singlet_audit(
    hamiltonian: np.ndarray,
    jumps: Sequence[tuple[np.ndarray, float]],
) -> SingletAudit:
    """Audit exact protection of ``P_-`` under one Lindblad generator."""

    h = np.asarray(hamiltonian, dtype=complex)
    if h.shape != (4, 4):
        raise ValueError("hamiltonian must act on two qubits")
    projector = singlet_projector()
    ket = singlet_ket()
    annihilation_terms = [
        np.sqrt(max(0.0, float(rate))) * np.asarray(jump, complex) @ ket
        for jump, rate in jumps
    ]
    annihilation = max(
        (float(np.linalg.norm(term)) for term in annihilation_terms), default=0.0
    )
    generator = liouvillian.lindbladian(h, jumps)
    return SingletAudit(
        pauli_decomposition_error=float(
            np.linalg.norm(projector - singlet_pauli_projector(), ord="fro")
        ),
        swap_antisymmetry_error=float(
            np.linalg.norm(swap_operator() @ ket + ket)
        ),
        hamiltonian_commutator_error=float(
            np.linalg.norm(h @ projector - projector @ h, ord="fro")
        ),
        jump_annihilation_error=annihilation,
        adjoint_invariance_error=float(
            np.linalg.norm(adjoint_action(generator, projector), ord="fro")
        ),
    )


def evolve_pair(
    first_state: np.ndarray,
    second_state: np.ndarray,
    h_static: np.ndarray,
    h_drive: np.ndarray,
    jumps: Sequence[tuple[np.ndarray, float]],
    suffix_inputs: Iterable[float],
    dt: float,
) -> list[dict[str, float | int]]:
    """Evolve two branches through an identical suffix and record ``<P_->``.

    Row zero is the prefix/suffix boundary.  Subsequent rows follow one common
    suffix input each.  The returned risk is the exact balanced error floor for
    every deterministic student whose representation is a function only of
    that suffix.
    """

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    rho_a = np.asarray(first_state, dtype=complex).copy()
    rho_b = np.asarray(second_state, dtype=complex).copy()
    if rho_a.shape != (4, 4) or rho_b.shape != (4, 4):
        raise ValueError("both branches must be two-qubit states")
    projector = singlet_projector()

    def row(step: int, suffix_input: float) -> dict[str, float | int]:
        first = expectation(rho_a, projector)
        second = expectation(rho_b, projector)
        bound = common_suffix_bound(first, second)
        return {
            "step": int(step),
            "time": float(step * dt),
            "suffix_input": float(suffix_input),
            "first_population": first,
            "second_population": second,
            "population_gap": bound.target_gap,
            "suffix_only_mse_floor": bound.minimum_mean_squared_error,
            "suffix_only_worst_case_floor": bound.minimum_worst_case_absolute_error,
            "first_negativity": negativity(rho_a),
            "second_negativity": negativity(rho_b),
        }

    records = [row(0, np.nan)]
    for step, value in enumerate(suffix_inputs, start=1):
        input_value = float(value)
        if not np.isfinite(input_value):
            raise ValueError("suffix inputs must be finite")
        generator = liouvillian.lindbladian(
            np.asarray(h_static, complex) + input_value * np.asarray(h_drive, complex),
            jumps,
        )
        rho_a = liouvillian.evolve_step(generator, rho_a, dt)
        rho_b = liouvillian.evolve_step(generator, rho_b, dt)
        records.append(row(step, input_value))
    return records


def symmetric_two_qubit_terms(coupling: float, field: float, drive: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the paper's symmetric two-qubit Ising and input-drive terms."""

    if not all(np.isfinite(value) for value in (coupling, field, drive)):
        raise ValueError("Hamiltonian parameters must be finite")
    couplings = np.array([[0.0, coupling], [coupling, 0.0]], dtype=float)
    h_static = reservoirs.ising_xx_hamiltonian(couplings, float(field), 2)
    h_drive = float(drive) * reservoirs.transverse_drive(2)
    return h_static, h_drive


def collective_jumps(gamma: float, asymmetry: float = 0.0) -> list[tuple[np.ndarray, float]]:
    """Return collective loss with coefficients proportional to ``(1, 1+eps)``."""

    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if not np.isfinite(asymmetry) or asymmetry <= -1.0:
        raise ValueError("asymmetry must be finite and greater than -1")
    return dissipators.collective_loss(2, float(gamma), c=(1.0, 1.0 + float(asymmetry)))


def asymmetric_singlet_initial_decay(gamma: float, asymmetry: float) -> float:
    """Return the exact initial ``-d<P_->/dt`` for an asymmetric collective jump."""

    jump, rate = collective_jumps(gamma, asymmetry)[0]
    ket = singlet_ket()
    return float(rate * np.vdot(jump @ ket, jump @ ket).real)
