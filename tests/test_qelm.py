import numpy as np
import pytest
from scipy.sparse.linalg import expm_multiply

from qrc_vs_qelm.matching import (
    assert_bundles_matched,
    assert_features_live,
    assert_matched,
)
from qrc_vs_qelm.core import dissipators, liouvillian, operators, readout, reservoirs
from qrc_vs_qelm.models.api import (
    QELMConfig,
    QRCConfig,
    build_features,
    build_qrc_features,
    windowed_qelm_from_qrc,
)
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


def test_parallel_encoder_uploads_each_lag_once_and_pads_unused_qubits():
    window = np.array([0.2, 0.7])
    encoded = qelm.encode_window(window, n_qubits=3, scheme="one_to_one")
    expected = operators.kron_list(
        [
            operators.input_state(0.2),
            operators.input_state(0.7),
            operators.input_state(0.0),
        ]
    )

    assert np.array_equal(encoded, expected)
    assert not np.array_equal(
        encoded,
        qelm.encode_window(window, n_qubits=3, scheme="cyclic"),
    )


def test_parallel_encoder_rejects_more_lags_than_qubits():
    with pytest.raises(ValueError, match="window length <= n_qubits"):
        qelm.encode_window(np.array([0.1, 0.2, 0.3]), 2, scheme="one_to_one")


def test_parallel_qelm_matches_manual_one_to_one_channel():
    n_qubits = 3
    d = 2
    inputs = np.linspace(0.05, 0.95, 9)
    obs = readout.pauli_observables(n_qubits, axes=("z",), max_weight=2)
    kwargs = dict(
        n_qubits=n_qubits,
        d=d,
        T_evolve=0.4,
        n_snapshots=2,
        gamma=0.0,
        dissipator="local_loss",
        J_s=0.7,
        h=0.3,
        coupling_seed=109,
        observables=obs,
    )

    public = qelm.build_qelm_features(inputs, encoding="parallel", **kwargs)
    rng = np.random.default_rng(kwargs["coupling_seed"])
    J = reservoirs.random_couplings(n_qubits, kwargs["J_s"], rng)
    H = reservoirs.ising_xx_hamiltonian(J, kwargs["h"], n_qubits)
    channel = qelm.QELMChannel.from_terms(
        n_qubits=n_qubits,
        H_static=H,
        jumps=[],
        T_evolve=kwargs["T_evolve"],
        n_snapshots=kwargs["n_snapshots"],
    )
    manual = channel.features(
        qelm.delay_embed(inputs, d),
        obs,
        encoding_scheme="one_to_one",
    )

    assert np.array_equal(public, manual)


def test_unitary_parallel_channel_matches_density_channel():
    n_qubits = 3
    rng = np.random.default_rng(811)
    J = reservoirs.random_couplings(n_qubits, 0.8, rng)
    H = reservoirs.ising_xx_hamiltonian(J, 0.4, n_qubits)
    obs = readout.pauli_observables(n_qubits, axes=("x", "y", "z"), max_weight=2)
    windows = np.array([[0.1, 0.4], [0.8, 0.3], [0.6, 0.9]])
    density = qelm.QELMChannel.from_terms(
        n_qubits=n_qubits,
        H_static=H,
        jumps=[],
        T_evolve=0.7,
        n_snapshots=3,
    )
    statevector = qelm.UnitaryQELMChannel.from_hamiltonian(
        n_qubits=n_qubits,
        H_static=H,
        T_evolve=0.7,
        n_snapshots=3,
    )

    expected = density.features(
        windows,
        obs,
        encoding_scheme="one_to_one",
    )
    actual = statevector.features(
        windows,
        obs,
        encoding_scheme="one_to_one",
    )

    assert np.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert np.all(actual >= -1.0 - 1e-12)
    assert np.all(actual <= 1.0 + 1e-12)


def test_time_delayed_qelm_memoryless_rows_with_window_longer_than_qubits():
    n_qubits = 3
    k = 6
    t0 = 11
    perturbed_idx = t0 - k
    u = np.linspace(0.03, 0.93, 16)
    u_changed = u.copy()
    u_changed[perturbed_idx] = 0.99

    kwargs = dict(
        n_qubits=n_qubits,
        d=k,
        T_evolve=0.25,
        n_snapshots=1,
        gamma=0.2,
        dissipator="local_loss",
        J_s=0.6,
        h=0.35,
        coupling_seed=7,
        encoding="time_delayed",
        quantize_levels=33,
    )
    X = qelm.build_qelm_features(u, **kwargs)
    X_changed = qelm.build_qelm_features(u_changed, **kwargs)

    assert X.shape == X_changed.shape
    assert np.array_equal(X[t0:], X_changed[t0:])
    for row in range(perturbed_idx, t0):
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


def test_time_delayed_qelm_deterministic_for_identical_args():
    rng = np.random.default_rng(61)
    u = rng.uniform(0.0, 1.0, size=12)
    kwargs = dict(
        n_qubits=3,
        d=7,
        T_evolve=0.3,
        n_snapshots=2,
        gamma=0.1,
        dissipator="dephasing",
        J_s=0.8,
        h=0.25,
        coupling_seed=13,
        encoding="time_delayed",
        quantize_levels=17,
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


@pytest.mark.parametrize("h_in", [0.5, 0.4])
def test_time_delayed_qelm_converges_to_qrc_when_window_covers_history(h_in):
    n_qubits = 3
    seed = 41
    n_snapshots = 2
    t0 = 9
    k_large = t0 + 1
    inputs = np.random.default_rng(1234).uniform(0.02, 0.98, size=14)
    obs = readout.pauli_observables(n_qubits, max_weight=2)
    qrc_cfg = QRCConfig(
        dt=0.25,
        h_in=h_in,
        n_virtual=n_snapshots,
        n_qubits=n_qubits,
        J_s=0.55,
        h=0.3,
        gamma=0.2,
        dissipator="local_loss",
        quantize_levels=17,
    )
    qrc = build_qrc_features(inputs, qrc_cfg, seed, observables=obs)
    small_cfg = windowed_qelm_from_qrc(qrc_cfg, window_k=3)
    large_cfg = windowed_qelm_from_qrc(qrc_cfg, window_k=k_large)
    assert_matched(qrc_cfg, large_cfg, observables=obs)

    small = build_features(small_cfg, inputs, seed, observables=obs).features
    large = build_features(large_cfg, inputs, seed, observables=obs).features

    large_dist = np.linalg.norm(large[t0] - qrc.features[t0])
    small_dist = np.linalg.norm(small[t0] - qrc.features[t0])
    assert large_dist < 1e-9
    assert large_dist <= small_dist + 1e-12


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


def test_time_delayed_qelm_gamma_zero_preserves_purity_and_real_bounded_features():
    n_qubits = 2
    seed = 23
    rng = np.random.default_rng(seed)
    J = reservoirs.random_couplings(n_qubits, J_s=0.5, rng=rng)
    H = reservoirs.ising_xx_hamiltonian(J, h=0.3, n_qubits=n_qubits)
    reservoir = reservoirs.LindbladReservoir.from_terms(
        n_qubits,
        H,
        reservoirs.transverse_drive(n_qubits),
        jumps=[],
        dt=0.2,
        cache_propagators=True,
        quantize=17,
    )
    states = qelm.time_delayed_window_states(
        reservoir,
        np.array([0.1, 0.4, 0.7, 0.9]),
        n_snapshots=3,
    )
    for rho in states:
        assert np.isclose(np.trace(rho @ rho), 1.0, atol=1e-9)

    obs = readout.pauli_observables(n_qubits, max_weight=2)
    X = qelm.build_qelm_features(
        np.linspace(0.05, 0.95, 8),
        n_qubits=n_qubits,
        d=5,
        T_evolve=0.2,
        n_snapshots=3,
        gamma=0.0,
        dissipator="local_loss",
        J_s=0.5,
        h=0.3,
        coupling_seed=seed,
        observables=obs,
        encoding="time_delayed",
        quantize_levels=17,
    )
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
            encoding="parallel",
        )


def test_qelm_config_accepts_canonical_parallel_encoding():
    cfg = QELMConfig(
        d=3,
        T_evolve=0.5,
        n_snapshots=2,
        n_qubits=4,
        J_s=0.7,
        h=0.4,
        gamma=0.0,
        dissipator="local_loss",
        encoding="parallel",
    )

    assert cfg.encoding == "parallel"


def test_qelm_config_allows_time_delayed_window_longer_than_qubit_register():
    cfg = QELMConfig(
        d=6,
        T_evolve=0.25,
        n_snapshots=1,
        n_qubits=3,
        J_s=0.7,
        h=0.4,
        gamma=0.0,
        dissipator="local_loss",
        encoding="time_delayed",
    )
    assert cfg.d == 6
    assert cfg.encoding == "time_delayed"

    X = qelm.build_qelm_features(
        np.linspace(0.1, 0.9, 8),
        n_qubits=cfg.n_qubits,
        d=cfg.d,
        T_evolve=cfg.T_evolve,
        n_snapshots=cfg.n_snapshots,
        gamma=cfg.gamma,
        dissipator=cfg.dissipator,
        J_s=cfg.J_s,
        h=cfg.h,
        coupling_seed=5,
        encoding=cfg.encoding,
    )
    assert X.shape[0] == 8


def test_qelm_config_rejects_unknown_encoding():
    with pytest.raises(ValueError, match="encoding"):
        QELMConfig(
            d=2,
            T_evolve=0.5,
            n_snapshots=1,
            n_qubits=3,
            J_s=0.7,
            h=0.4,
            gamma=0.0,
            dissipator="local_loss",
            encoding="temporal",
        )


def test_time_delayed_qelm_d_one_matches_qrc_first_reset_step():
    inputs = np.linspace(0.05, 0.95, 6)
    seed = 29
    qrc_cfg = QRCConfig(
        dt=0.2,
        h_in=0.5,
        n_virtual=2,
        n_qubits=2,
        J_s=0.6,
        h=0.35,
        gamma=0.1,
        dissipator="local_loss",
        quantize_levels=17,
    )
    qelm_cfg = windowed_qelm_from_qrc(qrc_cfg, window_k=1)

    qrc = build_features(qrc_cfg, inputs, seed)
    td_qelm = build_features(qelm_cfg, inputs, seed)

    assert td_qelm.first_valid == 0
    assert td_qelm.features.shape == qrc.features.shape
    assert np.all(np.isfinite(td_qelm.features))
    assert np.linalg.norm(td_qelm.features[0] - qrc.features[0]) < 1e-12


def test_time_delayed_qelm_zero_evolution_ignores_window_history():
    n_qubits = 2
    n_snapshots = 3
    obs = readout.pauli_observables(n_qubits, max_weight=2)
    X = qelm.build_qelm_features(
        np.linspace(0.05, 0.95, 7),
        n_qubits=n_qubits,
        d=4,
        T_evolve=0.0,
        n_snapshots=n_snapshots,
        gamma=0.2,
        dissipator="local_loss",
        J_s=0.6,
        h=0.35,
        coupling_seed=31,
        observables=obs,
        encoding="time_delayed",
        quantize_levels=17,
        h_in=0.5,
    )
    rho0 = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    rho0[0, 0] = 1.0
    expected = readout.features_from_states([rho0] * n_snapshots, obs).reshape(-1)

    assert np.allclose(X, expected)
    assert np.allclose(X, X[[0]])


def test_spatial_qelm_ignores_h_in_drive_parameter():
    inputs = np.linspace(0.04, 0.96, 10)
    kwargs = dict(
        n_qubits=3,
        d=3,
        T_evolve=0.4,
        n_snapshots=2,
        gamma=0.15,
        dissipator="local_loss",
        J_s=0.7,
        h=0.3,
        coupling_seed=43,
        encoding="spatial",
        quantize_levels=17,
    )

    no_drive = qelm.build_qelm_features(inputs, h_in=0.0, **kwargs)
    large_drive = qelm.build_qelm_features(inputs, h_in=3.0, **kwargs)

    assert np.array_equal(no_drive, large_drive)


def test_time_delayed_qelm_bundle_matching_preserves_readout_dimension():
    inputs = np.linspace(0.05, 0.95, 12)
    seed = 31
    qrc_cfg = QRCConfig(
        dt=0.3,
        h_in=0.5,
        n_virtual=2,
        n_qubits=3,
        J_s=0.7,
        h=0.4,
        gamma=0.2,
        dissipator="local_loss",
        quantize_levels=17,
    )
    qelm_cfg = windowed_qelm_from_qrc(qrc_cfg, window_k=6)
    assert_matched(qrc_cfg, qelm_cfg)

    qrc = build_features(qrc_cfg, inputs, seed)
    td_qelm = build_features(qelm_cfg, inputs, seed)

    assert td_qelm.first_valid == qelm.first_valid_index(qelm_cfg.d)
    assert qrc.features.shape[1] == td_qelm.features.shape[1]
    assert qrc.n_obs == td_qelm.n_obs
    assert qrc.n_multiplex == td_qelm.n_multiplex
    assert_bundles_matched(qrc, td_qelm)


def test_time_delayed_qelm_matched_bundle_passes_liveness_guard():
    inputs = np.random.default_rng(20240706).uniform(0.02, 0.98, size=80)
    seed = 37
    qrc_cfg = QRCConfig(
        dt=0.3,
        h_in=0.5,
        n_virtual=1,
        n_qubits=3,
        J_s=0.7,
        h=0.4,
        gamma=0.2,
        dissipator="local_loss",
        quantize_levels=17,
    )
    qelm_cfg = windowed_qelm_from_qrc(qrc_cfg, window_k=6)
    assert_matched(qrc_cfg, qelm_cfg)

    qrc = build_features(qrc_cfg, inputs, seed)
    td_qelm = build_features(qelm_cfg, inputs, seed)

    assert_bundles_matched(qrc, td_qelm)
    assert_features_live(qrc)
    assert_features_live(td_qelm)


def test_qelm_cyclic_encoding_repeats_window_across_qubits():
    n_qubits = 4
    rho = qelm.encode_window(np.array([0.25, 0.8]), n_qubits)
    rho0 = operators.partial_trace(rho, [0], n_qubits)
    rho1 = operators.partial_trace(rho, [1], n_qubits)
    rho2 = operators.partial_trace(rho, [2], n_qubits)
    rho3 = operators.partial_trace(rho, [3], n_qubits)
    assert np.allclose(rho0, rho2)
    assert np.allclose(rho1, rho3)
