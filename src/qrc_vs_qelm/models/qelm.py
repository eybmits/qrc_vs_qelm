"""Static, memoryless QELM feature map.

QELM rows are independent quantum feature-map evaluations on classical
delay-embedding windows. The Lindblad channel is input-independent, so snapshot
propagators are precomputed once per channel configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from qrc_vs_qelm.core import dissipators, liouvillian, operators, readout, reservoirs


def first_valid_index(d: int) -> int:
    """First row whose delay window contains no padding."""
    if d < 1:
        raise ValueError("d must be >= 1")
    return d - 1


def delay_embed(u: np.ndarray, d: int) -> np.ndarray:
    """Return causal delay windows ``(u[t], u[t-1], ..., u[t-d+1])``.

    The returned array has shape ``(len(u), d)``. Rows before
    :func:`first_valid_index` are zero-padded in unavailable lag positions; those
    rows are intentionally invalid for shared comparisons and should be excluded
    by the caller.
    """
    if d < 1:
        raise ValueError("d must be >= 1")
    values = np.asarray(u, dtype=float)
    if values.ndim != 1:
        raise ValueError("u must be a one-dimensional array")

    windows = np.zeros((len(values), d), dtype=float)
    for lag in range(d):
        if lag == 0:
            windows[:, lag] = values
        else:
            windows[lag:, lag] = values[:-lag]
    return windows


def encode_window(
    window: np.ndarray,
    n_qubits: int,
    scheme: str = "cyclic",
) -> np.ndarray:
    """Encode one delay window as a cyclic product-state density matrix."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if scheme != "cyclic":
        raise ValueError(f"unknown QELM encoding scheme: {scheme!r}")
    values = np.asarray(window, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("window must be a non-empty one-dimensional array")

    factors = [
        operators.input_state(float(values[q % len(values)]))
        for q in range(n_qubits)
    ]
    return operators.kron_list(factors)


@dataclass(frozen=True, init=False)
class QELMChannel:
    """Input-independent Lindblad channel with precomputed snapshot propagators."""

    n_qubits: int
    n_snapshots: int
    _L_base: np.ndarray
    _propagators: tuple[np.ndarray, ...]

    def __init__(
        self,
        n_qubits: int,
        n_snapshots: int,
        L_base: np.ndarray,
        propagators: Sequence[np.ndarray],
    ) -> None:
        object.__setattr__(self, "n_qubits", int(n_qubits))
        object.__setattr__(self, "n_snapshots", int(n_snapshots))
        object.__setattr__(
            self,
            "_L_base",
            np.array(L_base, dtype=complex, copy=True),
        )
        object.__setattr__(
            self,
            "_propagators",
            tuple(np.array(P, dtype=complex, copy=True) for P in propagators),
        )

    @property
    def L_base(self) -> np.ndarray:
        """Copy of the precomputed base Liouvillian."""
        return self._L_base.copy()

    @property
    def propagators(self) -> tuple[np.ndarray, ...]:
        """Copies of the precomputed snapshot propagators."""
        return tuple(P.copy() for P in self._propagators)

    @classmethod
    def from_terms(
        cls,
        n_qubits: int,
        H_static: np.ndarray,
        jumps: Sequence[tuple[np.ndarray, float]],
        T_evolve: float,
        n_snapshots: int,
    ) -> "QELMChannel":
        """Build the fixed Lindblad channel and all snapshot propagators."""
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if n_snapshots < 1:
            raise ValueError("n_snapshots must be >= 1")
        if T_evolve < 0:
            raise ValueError("T_evolve must be non-negative")

        dim = 2 ** n_qubits
        H = np.asarray(H_static, dtype=complex)
        if H.shape != (dim, dim):
            raise ValueError(
                f"H_static shape must be {(dim, dim)} for n_qubits={n_qubits}; "
                f"got {H.shape}"
            )
        checked_jumps: list[tuple[np.ndarray, float]] = []
        for idx, (jump, rate) in enumerate(jumps):
            L = np.asarray(jump, dtype=complex)
            if L.shape != (dim, dim):
                raise ValueError(
                    f"jump operator {idx} shape must be {(dim, dim)} "
                    f"for n_qubits={n_qubits}; got {L.shape}"
                )
            checked_jumps.append((L, float(rate)))

        L_base = liouvillian.lindbladian(H, checked_jumps)
        propagators = tuple(
            liouvillian.propagator(L_base, k * float(T_evolve) / n_snapshots)
            for k in range(1, n_snapshots + 1)
        )
        return cls(
            n_qubits=n_qubits,
            n_snapshots=n_snapshots,
            L_base=L_base,
            propagators=propagators,
        )

    def features(
        self,
        windows: np.ndarray,
        observables: Sequence[readout.Observable],
    ) -> np.ndarray:
        """Evaluate snapshot Pauli features for each independent delay window."""
        values = np.asarray(windows, dtype=float)
        if values.ndim != 2:
            raise ValueError("windows must be a two-dimensional array")
        obs = list(observables)
        n_obs = len(obs)
        X = np.empty((values.shape[0], self.n_snapshots * n_obs), dtype=float)

        for row_idx, window in enumerate(values):
            rho0 = encode_window(window, self.n_qubits)
            states = [
                liouvillian.apply_propagator(P, rho0)
                for P in self._propagators
            ]
            X[row_idx] = readout.features_from_states(states, obs).reshape(-1)
        return X


def _dissipator_jumps(
    name: str,
    n_qubits: int,
    gamma: float,
) -> list[tuple[np.ndarray, float]]:
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if gamma == 0:
        return []
    factories = {
        "local_loss": dissipators.local_loss,
        "collective_loss": dissipators.collective_loss,
        "dephasing": dissipators.dephasing,
    }
    try:
        factory = factories[name]
    except KeyError as exc:
        raise ValueError(f"unknown dissipator: {name!r}") from exc
    return factory(n_qubits, float(gamma))


def build_qelm_features(
    inputs: np.ndarray,
    *,
    n_qubits: int,
    d: int,
    T_evolve: float,
    n_snapshots: int,
    gamma: float,
    dissipator: str,
    J_s: float,
    h: float,
    coupling_seed: int,
    observables: Sequence[readout.Observable] | None = None,
) -> np.ndarray:
    """Build deterministic QELM features for a full input sequence."""
    rng = np.random.default_rng(coupling_seed)
    J = reservoirs.random_couplings(n_qubits, J_s, rng)
    H_static = reservoirs.ising_xx_hamiltonian(J, h, n_qubits)
    jumps = _dissipator_jumps(dissipator, n_qubits, float(gamma))
    obs = (
        readout.pauli_observables(n_qubits, axes=("x", "y", "z"), max_weight=2)
        if observables is None
        else list(observables)
    )
    channel = QELMChannel.from_terms(
        n_qubits=n_qubits,
        H_static=H_static,
        jumps=jumps,
        T_evolve=T_evolve,
        n_snapshots=n_snapshots,
    )
    return channel.features(delay_embed(inputs, d), obs)
