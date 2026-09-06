"""Exact local-Pauli backend for the separately frozen long-context follow-up.

This changes the numerical backend, not the QELM model: coupling generation,
the dense Hamiltonian, fixed dense unitaries, input upload, local XYZ observables,
and snapshot-major feature ordering are the canonical implementation's.  It
avoids constructing an unused product-state density matrix and contracts the
known sparse Pauli action rather than dense observable matrices for every row.

Only the prospective runner should opt in via ``efficient_qelm_backend()``.
Frozen studies and the installed package are not modified.  The scoped factory
patch is process-global and intended for the runner's single-threaded orchestration
(independent processes are safe), not overlapping contexts in different threads.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import re
from typing import Any

import numpy as np

from qrc_vs_qelm import canonical_study
from qrc_vs_qelm.core import readout, reservoirs
from qrc_vs_qelm.models import qelm


def _local_pauli_action(
    observables: Sequence[readout.Observable], n_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate actual operators, then return each row's column and coefficient.

    Validation is repeated on each call, so mutable Observable objects cannot
    silently change behind a cached name.  No dense matrix stack is allocated.
    Qubit zero is the most significant tensor-product bit, as in ``pauli_op``.
    """
    if not observables:
        raise ValueError("observables must be non-empty")
    dim = 2 ** n_qubits
    rows = np.arange(dim)
    columns = np.empty((len(observables), dim), dtype=np.intp)
    coefficients = np.empty((len(observables), dim), dtype=complex)
    for index, observable in enumerate(observables):
        matrix = np.asarray(observable.matrix, dtype=complex)
        if matrix.shape != (dim, dim):
            raise ValueError("observable shapes do not match n_qubits")
        match = re.fullmatch(r"([xyz])(\d+)", observable.name)
        if match is None or int(match[2]) >= n_qubits:
            raise ValueError("efficient backend supports declared local x/y/z observables only")
        axis, qubit = match[1], int(match[2])
        mask = 1 << (n_qubits - 1 - qubit)
        sign = 1 - 2 * ((rows & mask) != 0).astype(int)
        column = rows if axis == "z" else rows ^ mask
        coefficient = (
            np.ones(dim, dtype=complex) if axis == "x"
            else -1j * sign if axis == "y" else sign
        )
        if np.count_nonzero(matrix) != dim or not np.array_equal(
            matrix[rows, column], coefficient,
        ):
            raise ValueError("observable matrix does not equal its declared local Pauli")
        columns[index] = column
        coefficients[index] = coefficient
    return columns, coefficients


def _product_state(window: np.ndarray, n_qubits: int) -> np.ndarray:
    """Build the canonical one-to-one pure state without a density allocation."""
    encoded = np.zeros(n_qubits, dtype=float)
    encoded[:len(window)] = np.clip(window, 0.0, 1.0)
    state = np.ones(1, dtype=complex)
    for value in encoded:
        state = np.kron(state, [np.sqrt(1.0 - value), np.sqrt(value)])
    return state


class EfficientUnitaryQELMChannel(qelm.UnitaryQELMChannel):
    """Canonical channel with exact direct encoding and sparse expectations."""

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
        if encoding_scheme != "one_to_one":
            raise ValueError("efficient backend supports one_to_one encoding only")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if values.shape[1] == 0:
            raise ValueError("window must be a non-empty one-dimensional array")
        if values.shape[1] > self.n_qubits:
            raise ValueError("one_to_one encoding requires window length <= n_qubits")
        # Match input_state's endpoint tolerance and clipping; unlike that
        # function, explicitly fail closed on NaN rather than propagate it.
        if not np.all(np.isfinite(values)):
            raise ValueError("window inputs must be finite")
        if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
            raise ValueError("window inputs must lie in [0, 1]")
        obs = list(observables)
        columns, coefficients = _local_pauli_action(obs, self.n_qubits)
        features = np.empty((len(values), self.n_snapshots * len(obs)), dtype=float)
        for row_index, window in enumerate(values):
            initial = _product_state(window, self.n_qubits)
            for snapshot_index, unitary in enumerate(self._unitaries):
                state = unitary @ initial
                start = snapshot_index * len(obs)
                features[row_index, start:start + len(obs)] = np.real(np.sum(
                    state.conj()[None, :] * coefficients * state[columns], axis=1,
                ))
        return features


def efficient_student_channel(
    profile: dict[str, Any], coupling_seed: int,
) -> EfficientUnitaryQELMChannel:
    """Use exactly ``canonical_study._student_channel`` factory semantics."""
    config = profile["student"]
    n_qubits = int(config["n_qubits"])
    rng = np.random.default_rng(int(coupling_seed))
    couplings = reservoirs.random_couplings(n_qubits, float(config["J_s"]), rng)
    hamiltonian = reservoirs.ising_xx_hamiltonian(
        couplings, float(config["h"]), n_qubits,
    )
    return EfficientUnitaryQELMChannel.from_hamiltonian(
        n_qubits=n_qubits,
        H_static=hamiltonian,
        T_evolve=float(config["T_evolve"]),
        n_snapshots=int(profile["max_snapshots"]),
    )


@contextmanager
def efficient_qelm_backend() -> Iterator[None]:
    """Opt in for one study run; restore the previous factory even on failure."""
    previous = canonical_study._student_channel
    canonical_study._student_channel = efficient_student_channel
    try:
        yield
    finally:
        canonical_study._student_channel = previous
