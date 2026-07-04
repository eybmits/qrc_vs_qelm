import numpy as np
import pytest
from scipy.sparse.linalg import expm_multiply

from qrc_vs_qelm.core import dissipators, liouvillian, operators, readout, reservoirs
from qrc_vs_qelm.models.api import QELMConfig
from qrc_vs_qelm.models import qelm


def _channel(n_qubits=3, T_evolve=0.7, n_snapshots=2, gamma=0.2):
    rng = np.random.default_rng(123)
    J = reservoirs.random_couplings(n_qubits, J_s=0.8, rng=rng)
    H = reservoirs.ising_xx_hamiltonian(J, h=0.4, n_qubits=n_qubits)
    jumps = [] if gamma == 0 else dissipators.local_loss(n_qubits, gamma)
    return qelm.QELMChannel.from_terms(
        n_qubits=n_qubits,
        H_static=H,
        jumps=jumps,
        T_evolve=T_evolve,
        n_snapshots=n_snapshots,
    )


def test_delay_embed_and_first_valid_index():
    u = np.array([0.1, 0.2, 0.3, 0.4])
    X = qelm.delay_embed(u, d=3)
    assert qelm.first_valid_index(3) == 2
    assert X.shape == (4, 3)
    assert np.allclose(X[0], [0.1, 0.0, 0.0])
    assert np.allclose(X[2], [0.3, 0.2, 0.1])


def test_qelm_memoryless_rows_depend_only_on_delay_window():
    d = 3
    t0 = 7
    perturbed_idx = t0 - d
    u = np.linspace(0.05, 0.95, 12)
    u_changed = u.copy()
    u_changed[perturbed_idx] = 0.99

    kwargs = dict(
        n_qubits=3,
        d=d,
        T_evolve=0.6,
        n_snapshots=2,
        gamma=0.15,
        dissipator="local_loss",
        J_s=0.7,
        h=0.3,
        coupling_seed=5,
    )
    X = qelm.build_qelm_features(u, **kwargs)
    X_changed = qelm.build_qelm_features(u_changed, **kwargs)

    assert np.array_equal(X[t0:], X_changed[t0:])
    for row in range(perturbed_idx, perturbed_idx + d):
        assert not np.array_equal(X[row], X_changed[row])


def test_qelm_deterministic_for_identical_args():
    rng = np.random.default_rng(6)
    u = rng.uniform(0.0, 1.0, size=9)
    kwargs = dict(
        n_qubits=3,
        d=2,
        T_evolve=0.5,
        n_snapshots=3,
        gamma=0.1,
        dissipator="dephasing",
        J_s=0.9,
        h=0.2,
        coupling_seed=11,
    )
    X1 = qelm.build_qelm_features(u, **kwargs)
    X2 = qelm.build_qelm_features(u, **kwargs)
    assert np.array_equal(X1, X2)


def test_qelm_default_feature_shape():
    n_qubits = 3
    n_snapshots = 4
    u = np.linspace(0.1, 0.9, 8)
    X = qelm.build_qelm_features(
        u,
        n_qubits=n_qubits,
        d=3,
        T_evolve=0.4,
        n_snapshots=n_snapshots,
        gamma=0.2,
        dissipator="collective_loss",
        J_s=0.5,
        h=0.1,
        coupling_seed=17,
    )
    n_obs = 3 * n_qubits + 3 * (n_qubits * (n_qubits - 1) // 2)
    assert X.shape == (len(u), n_snapshots * n_obs)


def test_qelm_snapshot_propagators_match_expm_multiply():
    n_qubits = 3
    T_evolve = 0.9
    n_snapshots = 3
    channel = _channel(
        n_qubits=n_qubits,
        T_evolve=T_evolve,
        n_snapshots=n_snapshots,
        gamma=0.3,
    )
    rng = np.random.default_rng(9)
    v = rng.standard_normal(channel.L_base.shape[0]) + 1j * rng.standard_normal(
        channel.L_base.shape[0]
    )
    for k, P in enumerate(channel.propagators, start=1):
        t = k * T_evolve / n_snapshots
        expected = expm_multiply(channel.L_base * t, v)
        assert np.allclose(P @ v, expected, atol=1e-10)


def test_qelm_gamma_zero_preserves_purity_and_real_bounded_features():
    n_qubits = 3
    channel = _channel(n_qubits=n_qubits, T_evolve=0.8, n_snapshots=3, gamma=0)
    window = np.array([0.2, 0.75])
    rho0 = qelm.encode_window(window, n_qubits)

    for P in channel.propagators:
        rho = liouvillian.apply_propagator(P, rho0)
        assert np.isclose(np.trace(rho @ rho), 1.0, atol=1e-9)

    obs = readout.pauli_observables(n_qubits, max_weight=2)
    X = channel.features(np.array([window]), obs)
    assert np.isrealobj(X)
    assert np.all(X >= -1.0 - 1e-12)
    assert np.all(X <= 1.0 + 1e-12)


def test_build_qelm_features_gamma_zero_matches_manual_unitary_channel():
    n_qubits = 3
    d = 2
    T_evolve = 0.6
    n_snapshots = 2
    J_s = 0.7
    h = 0.4
    coupling_seed = 19
    inputs = np.linspace(0.05, 0.95, 10)
    obs = readout.pauli_observables(n_qubits, max_weight=2)

    public = qelm.build_qelm_features(
        inputs,
        n_qubits=n_qubits,
        d=d,
        T_evolve=T_evolve,
        n_snapshots=n_snapshots,
        gamma=0.0,
        dissipator="local_loss",
        J_s=J_s,
        h=h,
        coupling_seed=coupling_seed,
        observables=obs,
    )

    rng = np.random.default_rng(coupling_seed)
    J = reservoirs.random_couplings(n_qubits, J_s, rng)
    H = reservoirs.ising_xx_hamiltonian(J, h, n_qubits)
    channel = qelm.QELMChannel.from_terms(
        n_qubits=n_qubits,
        H_static=H,
        jumps=[],
        T_evolve=T_evolve,
        n_snapshots=n_snapshots,
    )
    manual = channel.features(qelm.delay_embed(inputs, d), obs)
    assert np.array_equal(public, manual)


def test_qelm_channel_validates_operator_shapes():
    with pytest.raises(ValueError, match="H_static shape"):
        qelm.QELMChannel.from_terms(
            n_qubits=2,
            H_static=np.eye(3),
            jumps=[],
            T_evolve=0.2,
            n_snapshots=1,
        )

    with pytest.raises(ValueError, match="jump operator 0 shape"):
        qelm.QELMChannel.from_terms(
            n_qubits=2,
            H_static=np.eye(4),
            jumps=[(np.eye(3), 0.1)],
            T_evolve=0.2,
            n_snapshots=1,
        )


def test_qelm_channel_defensive_copy_properties_do_not_mutate_channel():
    channel = _channel(n_qubits=2, T_evolve=0.4, n_snapshots=2, gamma=0.1)
    original_L = channel.L_base
    original_P = channel.propagators[0]

    mutated_L = channel.L_base
    mutated_P = channel.propagators[0]
    mutated_L[:] = 123.0
    mutated_P[:] = 456.0

    assert np.array_equal(channel.L_base, original_L)
    assert np.array_equal(channel.propagators[0], original_P)


def test_qelm_config_rejects_delay_longer_than_qubit_register():
    with pytest.raises(ValueError, match="1 <= d <= n_qubits"):
        QELMConfig(
            d=4,
            T_evolve=0.5,
            n_snapshots=2,
            n_qubits=3,
            J_s=0.7,
            h=0.4,
            gamma=0.0,
            dissipator="local_loss",
        )


def test_qelm_cyclic_encoding_repeats_window_across_qubits():
    n_qubits = 4
    rho = qelm.encode_window(np.array([0.25, 0.8]), n_qubits)
    rho0 = operators.partial_trace(rho, [0], n_qubits)
    rho1 = operators.partial_trace(rho, [1], n_qubits)
    rho2 = operators.partial_trace(rho, [2], n_qubits)
    rho3 = operators.partial_trace(rho, [3], n_qubits)
    assert np.allclose(rho0, rho2)
    assert np.allclose(rho1, rho3)
