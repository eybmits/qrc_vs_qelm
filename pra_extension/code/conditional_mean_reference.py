"""Finite-prefix conditional means for the fixed Lindblad teacher.

For iid inputs uniform on a finite alphabet, linearity gives the mean prefix
state as ``mean(Phi_a)**prefix_length(rho_initial)``. Applying a fixed suffix
and linear observables to that state gives the conditional *mean* output,
without sampling prefix branches. "Exact" here means exact for the declared
finite input law, up to the reservoir's numerical propagation error.

This reference is not a worst-case midpoint, a best feature/readout oracle,
or a calculation of conditional history variance. Those require different
information; in particular, a mean state alone does not supply second moments
of outputs across classical prefix histories.
"""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np

from qrc_vs_qelm.core.liouvillian import apply_propagator
from qrc_vs_qelm.core.readout import Observable
from qrc_vs_qelm.core.reservoirs import LindbladReservoir


def _dimension(reservoir: LindbladReservoir) -> int:
    if not isinstance(reservoir, LindbladReservoir):
        raise TypeError("reservoir must be a LindbladReservoir")
    if (
        isinstance(reservoir.n_qubits, (bool, np.bool_))
        or not isinstance(reservoir.n_qubits, Integral)
        or reservoir.n_qubits < 1
    ):
        raise ValueError("reservoir.n_qubits must be a positive integer")
    return 2 ** int(reservoir.n_qubits)


def _real_array(value: object, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real inputs")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real array") from exc
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _matrix(value: object, name: str, dimension: int) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a complex matrix") from exc
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def mean_prefix_state(
    reservoir: LindbladReservoir,
    alphabet: Sequence[float] | np.ndarray,
    prefix_length: int,
) -> np.ndarray:
    """Return the mean state after an iid uniform finite-alphabet prefix.

    ``alphabet`` is a nonempty 1-D sequence of distinct finite real symbols.
    The caller explicitly supplies the prefix law and length, e.g.
    ``np.linspace(0, 1, 33)`` and ``128`` for the follow-up protocol. The
    reservoir's own input snapping and propagation conventions are retained.
    A zero-length prefix returns a fresh copy of its initial state.

    When propagation caching is enabled, average the already cached channel
    matrices once and apply that average repeatedly, avoiding exponential
    branch enumeration and repeated matrix exponentials. Otherwise average
    ``reservoir.step`` over the alphabet at each prefix position. No cache
    option, model parameter, or random state is changed.
    """
    dimension = _dimension(reservoir)
    symbols = _real_array(alphabet, "alphabet", 1)
    if symbols.size == 0:
        raise ValueError("alphabet must be nonempty")
    if np.unique(symbols).size != symbols.size:
        raise ValueError("alphabet must contain distinct symbols")
    if (
        isinstance(prefix_length, (bool, np.bool_))
        or not isinstance(prefix_length, Integral)
        or prefix_length < 0
    ):
        raise ValueError("prefix_length must be a nonnegative integer")
    state = _matrix(reservoir.initial_state(), "initial state", dimension).copy()
    if prefix_length == 0:
        return state

    if reservoir.cache_propagators:
        # Mirror LindbladReservoir.step, including its snapping and cache keys.
        # Accumulate rather than stack all channels into another large array.
        mean_channel = np.zeros((dimension**2, dimension**2), dtype=complex)
        for symbol in symbols:
            channel = reservoir._sub_propagator(
                reservoir._snap(float(symbol)), reservoir.dt
            )
            if channel.shape != mean_channel.shape or not np.all(np.isfinite(channel)):
                raise ValueError("reservoir propagator has invalid shape or nonfinite values")
            mean_channel += channel / symbols.size
        for _ in range(int(prefix_length)):
            state = apply_propagator(mean_channel, state)
            _matrix(state, "propagated mean state", dimension)
    else:
        for _ in range(int(prefix_length)):
            next_state = np.zeros_like(state)
            for symbol in symbols:
                branch_state = _matrix(
                    reservoir.step(state, float(symbol)), "propagated state", dimension
                )
                next_state += branch_state / symbols.size
            state = _matrix(next_state, "propagated mean state", dimension)
    return state


def conditional_mean_outputs(
    reservoir: LindbladReservoir,
    mean_state: np.ndarray,
    windows: Sequence[Sequence[float]] | np.ndarray,
    observables: Sequence[Observable],
) -> np.ndarray:
    """Apply each suffix to ``mean_state`` and return its linear mean output.

    ``windows`` has shape ``(n_windows, suffix_length)`` in chronological
    (oldest-to-newest) order. Every suffix starts from an independent copy of
    the same mean state. The result has shape ``(n_windows, n_observables)``.
    Empty batches, zero-length suffixes, and empty observable panels are
    supported with these same shape conventions. Observables must be finite,
    dimension-matched Hermitian matrices wrapped as ``Observable`` objects.

    The caller supplies a valid density matrix; shape and finiteness are
    checked, but no projection, normalization, or positivity repair is done.
    This is a conditional mean only for the prefix law represented by
    ``mean_state`` and a fixed teacher with linear readout.
    """
    dimension = _dimension(reservoir)
    initial = _matrix(mean_state, "mean_state", dimension)
    suffixes = _real_array(windows, "windows", 2)
    matrices = []
    for index, observable in enumerate(observables):
        if not isinstance(observable, Observable):
            raise TypeError("observables must contain Observable objects")
        matrix = _matrix(observable.matrix, f"observable {index}", dimension)
        if not np.allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-12):
            raise ValueError(f"observable {index} must be Hermitian")
        matrices.append(matrix)

    outputs = np.empty((suffixes.shape[0], len(matrices)), dtype=float)
    for row, suffix in enumerate(suffixes):
        state = initial.copy()
        for symbol in suffix:
            state = _matrix(
                reservoir.step(state, float(symbol)), "propagated state", dimension
            )
        for column, matrix in enumerate(matrices):
            outputs[row, column] = float(np.real(np.trace(state @ matrix)))
    if not np.all(np.isfinite(outputs)):
        raise ValueError("conditional mean outputs contain nonfinite values")
    return outputs
