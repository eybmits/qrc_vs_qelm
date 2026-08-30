"""Windowed quantum feature maps.

Rows from ``spatial`` and ``parallel`` encodings are independent standard-QELM
evaluations: the complete delay vector is encoded before one fixed,
input-independent channel is applied. ``parallel`` assigns one lag to each
qubit and is the canonical TD-QELM used in the confirmatory emulation study.
The legacy ``time_delayed`` path instead resets and sequentially replays the
suffix through input-dependent QRC dynamics; scientifically it is a
memory-restricted reset-replay control rather than a standard QELM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import expm

from qrc_vs_qelm.core import dissipators, liouvillian, operators, readout, reservoirs


_VALID_ENCODINGS = {"spatial", "parallel", "time_delayed"}


def _validate_encoding(encoding: str) -> str:
    if encoding not in _VALID_ENCODINGS:
        raise ValueError(
            "encoding must be 'spatial', 'parallel', or 'time_delayed'"
        )
    return str(encoding)


def _cache_levels(levels: int | None) -> int | None:
    if levels is None or int(levels) == 0:
        return None
    if int(levels) != levels:
        raise ValueError("quantize_levels must be an integer, 0, or None")
    if int(levels) < 2:
        raise ValueError("quantize_levels must be >= 2, 0, or None")
    return int(levels)


def _quantize_inputs(u: np.ndarray, levels: int | None) -> np.ndarray:
    values = np.asarray(u, dtype=np.float64)
    checked = _cache_levels(levels)
    if checked is None:
        return values.copy()
    b = checked - 1
    return (np.round(values * b) / b).astype(np.float64, copy=False)


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
    """Encode one delay window as a product-state density matrix.

    ``cyclic`` preserves the repository's original spatial feature map by
    repeating window coordinates across the register. ``one_to_one`` is the
    canonical parallel delay encoding: coordinate ``j`` is uploaded to qubit
    ``j`` exactly once and any unused qubits remain in ``|0>``.  The latter is
    the architecture used by the canonical QRC-to-QELM emulation study.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if scheme not in {"cyclic", "one_to_one"}:
        raise ValueError(f"unknown QELM encoding scheme: {scheme!r}")
    values = np.asarray(window, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("window must be a non-empty one-dimensional array")
    if scheme == "one_to_one" and len(values) > n_qubits:
        raise ValueError(
            "one_to_one encoding requires window length <= n_qubits"
        )

    if scheme == "one_to_one":
        factors = [
            operators.input_state(float(values[q]) if q < len(values) else 0.0)
            for q in range(n_qubits)
        ]
    else:
        factors = [
            operators.input_state(float(values[q % len(values)]))
            for q in range(n_qubits)
        ]
    return operators.kron_list(factors)


def encode_window_statevector(
    window: np.ndarray,
    n_qubits: int,
    scheme: str = "one_to_one",
) -> np.ndarray:
    """Pure-state counterpart of :func:`encode_window` for unitary QELMs."""

    values = np.asarray(window, dtype=float)
    # Reuse all shape, range, and scheme validation from the density encoder.
    encode_window(values, n_qubits, scheme=scheme)
    clipped = np.clip(values, 0.0, 1.0)
    if scheme == "one_to_one":
        encoded = [float(clipped[q]) if q < len(clipped) else 0.0 for q in range(n_qubits)]
    else:
        encoded = [float(clipped[q % len(clipped)]) for q in range(n_qubits)]
    factors = [
        np.array([np.sqrt(1.0 - s), np.sqrt(s)], dtype=complex)
        for s in encoded
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
        *,
        encoding_scheme: str = "cyclic",
    ) -> np.ndarray:
        """Evaluate snapshot Pauli features for each independent delay window."""
        values = np.asarray(windows, dtype=float)
        if values.ndim != 2:
            raise ValueError("windows must be a two-dimensional array")
        obs = list(observables)
        n_obs = len(obs)
        X = np.empty((values.shape[0], self.n_snapshots * n_obs), dtype=float)

        for row_idx, window in enumerate(values):
            rho0 = encode_window(
                window,
                self.n_qubits,
                scheme=encoding_scheme,
            )
            states = [
                liouvillian.apply_propagator(P, rho0)
                for P in self._propagators
            ]
            X[row_idx] = readout.features_from_states(states, obs).reshape(-1)
        return X


@dataclass(frozen=True, init=False)
class UnitaryQELMChannel:
    """Statevector implementation of a fixed unitary standard-QELM channel.

    This avoids constructing a ``4**n`` Liouvillian and makes the canonical
    parallel student feasible at larger widths.  It is exactly a unitary
    special case of :class:`QELMChannel` up to floating-point roundoff.
    """

    n_qubits: int
    n_snapshots: int
    _hamiltonian: np.ndarray
    _unitaries: tuple[np.ndarray, ...]

    def __init__(
        self,
        n_qubits: int,
        n_snapshots: int,
        hamiltonian: np.ndarray,
        unitaries: Sequence[np.ndarray],
    ) -> None:
        object.__setattr__(self, "n_qubits", int(n_qubits))
        object.__setattr__(self, "n_snapshots", int(n_snapshots))
        object.__setattr__(
            self,
            "_hamiltonian",
            np.array(hamiltonian, dtype=complex, copy=True),
        )
        object.__setattr__(
            self,
            "_unitaries",
            tuple(np.array(U, dtype=complex, copy=True) for U in unitaries),
        )

    @property
    def hamiltonian(self) -> np.ndarray:
        return self._hamiltonian.copy()

    @property
    def unitaries(self) -> tuple[np.ndarray, ...]:
        return tuple(U.copy() for U in self._unitaries)

    @classmethod
    def from_hamiltonian(
        cls,
        *,
        n_qubits: int,
        H_static: np.ndarray,
        T_evolve: float,
        n_snapshots: int,
    ) -> "UnitaryQELMChannel":
        if int(n_qubits) < 1:
            raise ValueError("n_qubits must be >= 1")
        if int(n_snapshots) < 1:
            raise ValueError("n_snapshots must be >= 1")
        if float(T_evolve) < 0.0:
            raise ValueError("T_evolve must be non-negative")
        dim = 2 ** int(n_qubits)
        H = np.asarray(H_static, dtype=complex)
        if H.shape != (dim, dim):
            raise ValueError(
                f"H_static shape must be {(dim, dim)} for n_qubits={n_qubits}; "
                f"got {H.shape}"
            )
        if not np.allclose(H, H.conj().T, atol=1e-12, rtol=0.0):
            raise ValueError("H_static must be Hermitian")
        unitaries = tuple(
            expm(-1j * H * (k * float(T_evolve) / int(n_snapshots)))
            for k in range(1, int(n_snapshots) + 1)
        )
        return cls(
            n_qubits=int(n_qubits),
            n_snapshots=int(n_snapshots),
            hamiltonian=H,
            unitaries=unitaries,
        )

    def features(
        self,
        windows: np.ndarray,
        observables: Sequence[readout.Observable],
        *,
        encoding_scheme: str = "one_to_one",
    ) -> np.ndarray:
        values = np.asarray(windows, dtype=float)
        if values.ndim != 2:
            raise ValueError("windows must be a two-dimensional array")
        obs = list(observables)
        if not obs:
            raise ValueError("observables must be non-empty")
        matrices = np.stack([np.asarray(item.matrix, dtype=complex) for item in obs])
        dim = 2 ** self.n_qubits
        if matrices.shape[1:] != (dim, dim):
            raise ValueError("observable shapes do not match n_qubits")
        X = np.empty(
            (values.shape[0], self.n_snapshots * len(obs)),
            dtype=float,
        )
        for row_idx, window in enumerate(values):
            psi0 = encode_window_statevector(
                window,
                self.n_qubits,
                scheme=encoding_scheme,
            )
            for snapshot_idx, U in enumerate(self._unitaries):
                psi = U @ psi0
                start = snapshot_idx * len(obs)
                X[row_idx, start : start + len(obs)] = np.real(
                    np.einsum("i,kij,j->k", psi.conj(), matrices, psi)
                )
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


def time_delayed_window_states(
    reservoir: reservoirs.LindbladReservoir,
    window: np.ndarray,
    n_snapshots: int,
) -> list[np.ndarray]:
    """Replay one causal window from reset and return readout states.

    ``window`` is ordered oldest to newest. Snapshot handling follows the QRC
    virtual-node convention: the first ``k - 1`` inputs are full reservoir
    steps, then ``n_snapshots`` states are sampled inside the newest input step.
    This keeps the feature width identical to spatial QELM,
    ``n_snapshots * n_observables``.
    """
    values = np.asarray(window, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("window must be a non-empty one-dimensional array")
    if int(n_snapshots) < 1:
        raise ValueError("n_snapshots must be >= 1")

    rho = reservoir.initial_state()
    for s in values[:-1]:
        rho = reservoir.step(rho, float(s))
    return reservoir.substep_states(rho, float(values[-1]), int(n_snapshots))


def _time_delayed_features(
    inputs: np.ndarray,
    *,
    reservoir: reservoirs.LindbladReservoir,
    d: int,
    n_snapshots: int,
    observables: Sequence[readout.Observable],
) -> np.ndarray:
    windows_newest_first = delay_embed(inputs, d)
    obs = list(observables)
    n_obs = len(obs)
    X = np.empty((windows_newest_first.shape[0], int(n_snapshots) * n_obs), dtype=float)

    for row_idx, newest_first in enumerate(windows_newest_first):
        states = time_delayed_window_states(
            reservoir,
            newest_first[::-1],
            int(n_snapshots),
        )
        X[row_idx] = readout.features_from_states(states, obs).reshape(-1)
    return X


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
    encoding: str = "spatial",
    quantize_levels: int | None = None,
    h_in: float = 1.0,
) -> np.ndarray:
    """Build deterministic QELM features for a full input sequence."""
    checked_encoding = _validate_encoding(encoding)
    if d < 1:
        raise ValueError("d must be >= 1")
    if T_evolve < 0:
        raise ValueError("T_evolve must be non-negative")
    if n_snapshots < 1:
        raise ValueError("n_snapshots must be >= 1")

    driven_inputs = _quantize_inputs(inputs, quantize_levels)
    quantize = _cache_levels(quantize_levels)
    rng = np.random.default_rng(coupling_seed)
    J = reservoirs.random_couplings(n_qubits, J_s, rng)
    H_static = reservoirs.ising_xx_hamiltonian(J, h, n_qubits)
    jumps = _dissipator_jumps(dissipator, n_qubits, float(gamma))
    obs = (
        readout.pauli_observables(n_qubits, axes=("x", "y", "z"), max_weight=2)
        if observables is None
        else list(observables)
    )
    if checked_encoding == "time_delayed":
        reservoir = reservoirs.LindbladReservoir.from_terms(
            n_qubits,
            H_static,
            float(h_in) * reservoirs.transverse_drive(n_qubits),
            jumps,
            T_evolve,
            cache_propagators=quantize is not None,
            quantize=quantize,
        )
        return _time_delayed_features(
            driven_inputs,
            reservoir=reservoir,
            d=d,
            n_snapshots=n_snapshots,
            observables=obs,
        )

    channel = QELMChannel.from_terms(
        n_qubits=n_qubits,
        H_static=H_static,
        jumps=jumps,
        T_evolve=T_evolve,
        n_snapshots=n_snapshots,
    )
    return channel.features(
        delay_embed(driven_inputs, d),
        obs,
        encoding_scheme=(
            "one_to_one" if checked_encoding == "parallel" else "cyclic"
        ),
    )
