from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "pra_extension" / "code", REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from conditional_mean_reference import (  # noqa: E402
    conditional_mean_outputs,
    mean_prefix_state,
)
from qrc_vs_qelm.core import operators, readout  # noqa: E402
from qrc_vs_qelm.core.reservoirs import LindbladReservoir  # noqa: E402


def _reservoir(n_qubits: int = 1, cached: bool = True, quantize=None):
    h_static = 0.37 * operators.pauli_op("z", 0, n_qubits)
    h_drive = 0.91 * operators.pauli_op("x", 0, n_qubits)
    if n_qubits == 2:
        h_static += 0.23 * operators.two_site_pauli("x", 0, 1, n_qubits)
        h_static += 0.19 * operators.pauli_op("y", 1, n_qubits)
    return LindbladReservoir.from_terms(
        n_qubits,
        h_static,
        h_drive,
        [(operators.sminus(0, n_qubits), 0.17)],
        dt=0.63,
        cache_propagators=cached,
        quantize=quantize,
    )


def _enumerated_states(reservoir, alphabet, length):
    states = []
    for prefix in product(alphabet, repeat=length):
        state = reservoir.initial_state()
        for symbol in prefix:
            state = reservoir.step(state, float(symbol))
        states.append(state)
    return np.asarray(states)


@pytest.mark.parametrize("n_qubits", [1, 2])
@pytest.mark.parametrize("cached", [False, True])
def test_mean_prefix_matches_exhaustive_noncommuting_channels(n_qubits, cached):
    reservoir = _reservoir(n_qubits, cached)
    alphabet = [0.0, 1.0]
    initial = reservoir.initial_state()
    forward = reservoir.step(reservoir.step(initial, 0.0), 1.0)
    reverse = reservoir.step(reservoir.step(initial, 1.0), 0.0)
    assert np.linalg.norm(forward - reverse) > 0.05

    expected = _enumerated_states(reservoir, alphabet, 3).mean(axis=0)
    actual = mean_prefix_state(reservoir, alphabet, 3)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(actual, actual.conj().T, atol=2e-13)
    np.testing.assert_allclose(np.trace(actual), 1.0, atol=2e-13)


@pytest.mark.parametrize("cached", [False, True])
def test_conditional_outputs_match_average_over_individual_histories(cached):
    reservoir = _reservoir(2, cached)
    alphabet = np.asarray([0.0, 0.4, 1.0])
    windows = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.4, 0.4]])
    observables = readout.pauli_observables(2, axes=("x", "z"), max_weight=2)
    observables += [readout.Observable(
        "linear combination", 0.7 * observables[0].matrix - 0.3 * observables[-1].matrix
    )]
    branches = _enumerated_states(reservoir, alphabet, 2)
    expected = []
    for window in windows:
        evolved = []
        for branch in branches:
            state = branch.copy()
            for symbol in window:
                state = reservoir.step(state, float(symbol))
            evolved.append(state)
        expected.append(readout.features_from_states(evolved, observables).mean(axis=0))

    mean_state = mean_prefix_state(reservoir, alphabet, 2)
    saved_state = mean_state.copy()
    saved_windows = windows.copy()
    actual = conditional_mean_outputs(reservoir, mean_state, windows, observables)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_array_equal(mean_state, saved_state)
    np.testing.assert_array_equal(windows, saved_windows)
    np.testing.assert_array_equal(
        actual, conditional_mean_outputs(reservoir, mean_state, windows, observables)
    )


def test_mean_is_not_a_worst_case_midpoint():
    reservoir = _reservoir()
    branches = _enumerated_states(reservoir, [0.0, 1.0], 3)
    observables = [readout.Observable("z", operators.SZ)]
    values = readout.features_from_states(branches, observables)[:, 0]
    actual = conditional_mean_outputs(
        reservoir, mean_prefix_state(reservoir, [0.0, 1.0], 3),
        np.empty((1, 0)), observables,
    )[0, 0]
    np.testing.assert_allclose(actual, values.mean(), atol=2e-13)
    assert abs(actual - 0.5 * (values.min() + values.max())) > 1e-3


def test_zero_prefix_empty_suffix_and_empty_batches():
    reservoir = _reservoir()
    mean_state = mean_prefix_state(reservoir, [0.0, 1.0], 0)
    np.testing.assert_array_equal(mean_state, reservoir.initial_state())
    assert reservoir._prop_cache == {}
    observables = readout.pauli_observables(1, max_weight=1)
    actual = conditional_mean_outputs(reservoir, mean_state, np.empty((3, 0)), observables)
    expected = readout.features_from_states([mean_state] * 3, observables)
    np.testing.assert_array_equal(actual, expected)
    assert conditional_mean_outputs(
        reservoir, mean_state, np.empty((0, 2)), observables
    ).shape == (0, 3)
    assert conditional_mean_outputs(
        reservoir, mean_state, np.empty((2, 0)), []
    ).shape == (2, 0)


def test_cached_channel_reuse_and_reservoir_snapping():
    reservoir = _reservoir(quantize=3)
    alphabet = [0.1, 0.4, 0.9]
    expected = _enumerated_states(reservoir, alphabet, 2).mean(axis=0)
    cached_channels = dict(reservoir._prop_cache)
    actual = mean_prefix_state(reservoir, alphabet, 2)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    assert reservoir._prop_cache.keys() == cached_channels.keys()
    assert all(reservoir._prop_cache[key] is value for key, value in cached_channels.items())
    np.testing.assert_array_equal(actual, mean_prefix_state(reservoir, alphabet, 2))


def test_single_symbol_long_prefix_matches_direct_iteration():
    reservoir = _reservoir()
    state = reservoir.initial_state()
    for _ in range(128):
        state = reservoir.step(state, 0.4)
    np.testing.assert_allclose(mean_prefix_state(reservoir, [0.4], 128), state, atol=2e-13)


@pytest.mark.parametrize("alphabet", [[], [[0.0, 1.0]], [np.nan], [np.inf],
                                         [0.0, 0.0], [0.0 + 1.0j]])
def test_invalid_alphabet_is_rejected(alphabet):
    with pytest.raises(ValueError):
        mean_prefix_state(_reservoir(), alphabet, 1)


@pytest.mark.parametrize("length", [-1, 1.5, True, np.bool_(False), "2"])
def test_invalid_prefix_length_is_rejected(length):
    with pytest.raises(ValueError, match="prefix_length"):
        mean_prefix_state(_reservoir(), [0.0, 1.0], length)


@pytest.mark.parametrize("state", [np.eye(3), np.ones(4), np.full((2, 2), np.nan),
                                      np.full((2, 2), np.inf)])
def test_invalid_mean_state_is_rejected(state):
    with pytest.raises(ValueError, match="mean_state"):
        conditional_mean_outputs(_reservoir(), state, np.empty((1, 0)), [])


@pytest.mark.parametrize("windows", [[0.0, 1.0], [[np.nan]], [[np.inf]], [[1j]],
                                        [[0.0], [0.0, 1.0]]])
def test_invalid_windows_are_rejected(windows):
    reservoir = _reservoir()
    with pytest.raises(ValueError):
        conditional_mean_outputs(reservoir, reservoir.initial_state(), windows, [])


@pytest.mark.parametrize("matrix", [np.eye(3), np.full((2, 2), np.nan),
                                       np.full((2, 2), np.inf), operators.SMINUS])
def test_invalid_observable_is_rejected(matrix):
    reservoir = _reservoir()
    with pytest.raises(ValueError, match="observable"):
        conditional_mean_outputs(
            reservoir, reservoir.initial_state(), np.empty((1, 0)),
            [readout.Observable("invalid", matrix)],
        )


def test_input_types_are_explicit():
    with pytest.raises(TypeError, match="LindbladReservoir"):
        mean_prefix_state(object(), [0.0], 1)
    reservoir = _reservoir()
    with pytest.raises(TypeError, match="Observable"):
        conditional_mean_outputs(
            reservoir, reservoir.initial_state(), np.empty((1, 0)), [operators.SZ]
        )
