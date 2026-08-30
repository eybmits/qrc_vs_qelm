from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "pra_extension" / "code"
SRC_ROOT = REPO_ROOT / "src"
for path in (CODE_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from holevo_bridge import (  # noqa: E402
    apply_transfer,
    coherence_sector_bridge,
    computational_dephasing,
    conditional_holevo_nats,
    holevo_risk_bridge,
    observable_history_risk,
    von_neumann_entropy,
)
from qrc_vs_qelm.core import liouvillian  # noqa: E402


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def ket_density(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, complex)
    return np.outer(value, value.conj())


def unitary_transfer(unitary: np.ndarray) -> np.ndarray:
    return np.kron(unitary.conj(), unitary)


def test_entropy_and_holevo_for_orthogonal_pure_states() -> None:
    states = np.stack(
        [ket_density(np.array([1, 0])), ket_density(np.array([0, 1]))]
    )
    assert von_neumann_entropy(states[0]) == 0.0
    np.testing.assert_allclose(conditional_holevo_nats(states), np.log(2), atol=1e-14)


def test_observable_risk_certifies_conditional_holevo_memory() -> None:
    states = np.stack(
        [ket_density(np.array([1, 0])), ket_density(np.array([0, 1]))]
    )
    bridge = holevo_risk_bridge(states, [Z])
    np.testing.assert_allclose(bridge.history_risk, 1.0, atol=1e-14)
    np.testing.assert_allclose(bridge.holevo_nats, np.log(2), atol=1e-14)
    assert bridge.history_risk <= bridge.pinsker_upper_bound
    np.testing.assert_allclose(bridge.certified_holevo_lower_bound, 0.5)


def test_random_ensembles_obey_holevo_pinsker_bridge() -> None:
    rng = np.random.default_rng(28_082_901)
    observables = [X, Z, (X + Z) / np.sqrt(2)]
    for n_states in (2, 3, 7):
        states = []
        for _ in range(n_states):
            matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            rho = matrix @ matrix.conj().T
            states.append(rho / np.trace(rho))
        probabilities = rng.random(n_states)
        bridge = holevo_risk_bridge(
            np.stack(states), observables, probabilities=probabilities
        )
        assert bridge.pinsker_slack >= 0
        assert bridge.history_risk <= 2 * bridge.holevo_nats + 2e-9


def test_vanishing_holevo_removes_every_observable_history_floor() -> None:
    state = np.array([[0.7, 0.1j], [-0.1j, 0.3]], dtype=complex)
    states = np.repeat(state[None, :, :], 5, axis=0)
    assert conditional_holevo_nats(states) < 1e-14
    assert observable_history_risk(states, [X, Z]) < 1e-28


def test_boundary_dephasing_is_trace_preserving_and_idempotent() -> None:
    plus = ket_density(np.array([1, 1]) / np.sqrt(2))
    dephased = computational_dephasing(plus)
    np.testing.assert_allclose(dephased, I2 / 2, atol=1e-14)
    np.testing.assert_allclose(computational_dephasing(dephased), dephased)


def test_coherence_only_history_is_converted_to_population_signal() -> None:
    plus = ket_density(np.array([1, 1]) / np.sqrt(2))
    minus = ket_density(np.array([1, -1]) / np.sqrt(2))
    states = np.stack([plus, minus])
    bridge = coherence_sector_bridge(
        states, unitary_transfer(HADAMARD), [Z]
    )
    np.testing.assert_allclose(bridge.history_risk, 1.0, atol=1e-13)
    np.testing.assert_allclose(bridge.population_history_risk, 0.0, atol=1e-13)
    np.testing.assert_allclose(bridge.coherence_history_risk, 1.0, atol=1e-13)
    np.testing.assert_allclose(bridge.population_coherence_cross_term, 0.0, atol=1e-13)
    np.testing.assert_allclose(bridge.dephasing_intervention_mse, 1.0, atol=1e-13)


def test_population_only_history_has_zero_coherence_sector() -> None:
    states = np.stack(
        [ket_density(np.array([1, 0])), ket_density(np.array([0, 1]))]
    )
    bridge = coherence_sector_bridge(states, np.eye(4, dtype=complex), [Z])
    np.testing.assert_allclose(bridge.history_risk, 1.0, atol=1e-14)
    np.testing.assert_allclose(bridge.population_history_risk, 1.0, atol=1e-14)
    np.testing.assert_allclose(bridge.coherence_history_risk, 0.0, atol=1e-14)


def test_transfer_uses_repository_column_vectorization() -> None:
    plus = ket_density(np.array([1, 1]) / np.sqrt(2))
    evolved = apply_transfer(plus[None, :, :], unitary_transfer(HADAMARD))[0]
    expected = HADAMARD @ plus @ HADAMARD.conj().T
    np.testing.assert_allclose(evolved, expected, atol=1e-14)
    np.testing.assert_allclose(
        liouvillian.vec(evolved), unitary_transfer(HADAMARD) @ liouvillian.vec(plus)
    )
