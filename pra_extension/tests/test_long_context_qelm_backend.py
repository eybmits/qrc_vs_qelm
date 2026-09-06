"""Development-only equivalence probes; no prospective study seeds or cells."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from long_context_qelm_backend import (  # noqa: E402
    EfficientUnitaryQELMChannel,
    efficient_qelm_backend,
    efficient_student_channel,
)
from qrc_vs_qelm import canonical_study  # noqa: E402
from qrc_vs_qelm.core import readout  # noqa: E402
from qrc_vs_qelm.models import qelm  # noqa: E402


def _profile(n_qubits: int, snapshots: int = 4) -> dict:
    return {
        "student": {"n_qubits": n_qubits, "J_s": 1.0, "h": 0.5, "T_evolve": 0.5},
        "max_snapshots": snapshots,
    }


@pytest.fixture(autouse=True)
def _single_threaded_blas():
    # Scheduling only; this does not change the channel or its floating dtype.
    with threadpool_limits(limits=1):
        yield


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_randomized_equivalence_and_canonical_factory(n_qubits):
    profile = _profile(n_qubits)
    original = canonical_study._student_channel(profile, 1700010 + n_qubits)
    efficient = efficient_student_channel(profile, 1700010 + n_qubits)
    np.testing.assert_array_equal(efficient.hamiltonian, original.hamiltonian)
    for actual, expected in zip(efficient.unitaries, original.unitaries, strict=True):
        np.testing.assert_array_equal(actual, expected)
    rng = np.random.default_rng(1700020 + n_qubits)
    obs = readout.pauli_observables(n_qubits, max_weight=1)
    for width in range(1, n_qubits + 1):
        values = rng.uniform(size=(5, width))
        actual = efficient.features(values, obs)
        expected = original.features(values, obs)
        assert actual.shape == (5, 4 * 3 * n_qubits)
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
        # Subset/reordered lists must preserve their declared order as well.
        reordered = obs[::-1][::2]
        np.testing.assert_allclose(
            efficient.features(values, reordered), original.features(values, reordered),
            atol=1e-11, rtol=0,
        )


@pytest.fixture(scope="module")
def ten_qubit_channels():
    # Explicitly reserved development seed, never a fresh study coupling seed.
    with threadpool_limits(limits=1):
        efficient = efficient_student_channel(_profile(10), 1700002)
    original = qelm.UnitaryQELMChannel(
        n_qubits=10, n_snapshots=4, hamiltonian=efficient._hamiltonian,
        unitaries=efficient._unitaries,
    )
    return original, efficient, readout.pauli_observables(10, max_weight=1)


@pytest.mark.parametrize("width", [2, 10])
def test_ten_qubit_all_snapshot_endpoint_equal_value_and_determinism_probes(
    width, ten_qubit_channels,
):
    original, efficient, obs = ten_qubit_channels
    random_row = np.random.default_rng(1700003).uniform(size=(1, width))
    values = np.vstack((random_row, np.zeros((1, width)), np.ones((1, width)),
                        np.full((1, width), 0.5), np.full((1, width), 0.37)))
    expected = original.features(values, obs)
    actual = efficient.features(values, obs)
    assert actual.shape == (5, 120)
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
    for snapshots in (1, 2, 4):
        np.testing.assert_allclose(
            actual[:, :30 * snapshots], expected[:, :30 * snapshots], atol=1e-11, rtol=0,
        )
    np.testing.assert_array_equal(efficient.features(values, obs), actual)


def test_context_restores_factory_on_success_failure_and_nesting():
    original = canonical_study._student_channel
    with efficient_qelm_backend():
        assert canonical_study._student_channel is efficient_student_channel
        assert isinstance(canonical_study._student_channel(_profile(1), 1700004),
                          EfficientUnitaryQELMChannel)
        with efficient_qelm_backend():
            assert canonical_study._student_channel is efficient_student_channel
        assert canonical_study._student_channel is efficient_student_channel
    assert canonical_study._student_channel is original
    with pytest.raises(RuntimeError, match="probe failure"):
        with efficient_qelm_backend():
            raise RuntimeError("probe failure")
    assert canonical_study._student_channel is original


@pytest.fixture
def small_channel():
    return efficient_student_channel(_profile(2), 1700005)


@pytest.mark.parametrize("values,match", [
    (np.zeros(2), "two-dimensional"),
    (np.zeros((1, 3)), "window length"),
    (np.zeros((1, 0)), "non-empty"),
    (np.array([[np.nan]]), "finite"),
    (np.array([[np.inf]]), "finite"),
    (np.array([[-1e-6]]), r"\[0, 1\]"),
    (np.array([[1.000001]]), r"\[0, 1\]"),
])
def test_input_validation(small_channel, values, match):
    with pytest.raises(ValueError, match=match):
        small_channel.features(values, readout.pauli_observables(2, max_weight=1))


def test_endpoint_tolerance_empty_rows_and_rejection_of_other_encodings(small_channel):
    obs = readout.pauli_observables(2, max_weight=1)
    values = np.array([[-5e-13, 1.0 + 5e-13]])
    original = canonical_study._student_channel(_profile(2), 1700005)
    np.testing.assert_allclose(small_channel.features(values, obs),
                               original.features(values, obs), atol=1e-11, rtol=0)
    assert small_channel.features(np.empty((0, 2)), obs).shape == (0, 24)
    with pytest.raises(ValueError, match="one_to_one"):
        small_channel.features(np.zeros((1, 1)), obs, encoding_scheme="cyclic")


def test_validates_actual_operator_not_just_its_name_and_rechecks_mutations(small_channel):
    values = np.array([[0.2, 0.7]])
    obs = readout.pauli_observables(2, max_weight=1)
    small_channel.features(values, obs)
    obs[0].matrix[0, 0] = 1e-12
    with pytest.raises(ValueError, match="declared local Pauli"):
        small_channel.features(values, obs)
    forged = [readout.Observable("x0", readout.pauli_observables(2, axes=["y"])[0].matrix)]
    with pytest.raises(ValueError, match="declared local Pauli"):
        small_channel.features(values, forged)
    with pytest.raises(ValueError, match="local x/y/z"):
        small_channel.features(values, readout.pauli_observables(2, max_weight=2))
    with pytest.raises(ValueError, match="shapes"):
        small_channel.features(values, readout.pauli_observables(1, max_weight=1))
    with pytest.raises(ValueError, match="non-empty"):
        small_channel.features(values, [])


def test_fast_path_does_not_call_density_or_original_state_encoder(small_channel, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dense encoder was called")
    monkeypatch.setattr(qelm, "encode_window", forbidden)
    monkeypatch.setattr(qelm, "encode_window_statevector", forbidden)
    actual = small_channel.features(
        np.array([[0.2, 0.8]]), readout.pauli_observables(2, max_weight=1),
    )
    assert actual.shape == (1, 24)
    assert np.all(np.isfinite(actual))
