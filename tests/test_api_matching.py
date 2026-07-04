from dataclasses import replace

import numpy as np
import pytest

from qrc_vs_qelm.core import readout
from qrc_vs_qelm.matching import (
    MatchingError,
    assert_bundles_matched,
    assert_features_live,
    assert_matched,
    effective_dimension,
    multiplex_efficacy,
    physical_shot_budget,
    recommend_T_evolve,
)
from qrc_vs_qelm.models.api import (
    FeatureBundle,
    QELMConfig,
    QRCConfig,
    build_features,
    build_qrc_features,
    quantize_inputs,
)


def _matched_configs():
    qrc_cfg = QRCConfig(
        dt=0.2,
        h_in=0.3,
        n_virtual=2,
        n_qubits=2,
        J_s=0.7,
        h=0.4,
        gamma=0.2,
        dissipator="local_loss",
        quantize_levels=11,
    )
    qelm_cfg = QELMConfig(
        d=2,
        T_evolve=0.5,
        n_snapshots=2,
        n_qubits=2,
        J_s=0.7,
        h=0.4,
        gamma=0.2,
        dissipator="local_loss",
        quantize_levels=11,
    )
    return qrc_cfg, qelm_cfg


def test_matched_builders_have_same_shape_couplings_and_dimension():
    qrc_cfg, qelm_cfg = _matched_configs()
    inputs = np.linspace(0.05, 0.95, 24)
    seed = 23

    qrc = build_features(qrc_cfg, inputs, seed)
    qelm = build_features(qelm_cfg, inputs, seed)

    assert qrc.features.shape == qelm.features.shape
    assert np.array_equal(qrc.couplings, qelm.couplings)
    assert qrc.features.shape[1] == qrc.n_obs * qrc.n_multiplex
    assert qelm.features.shape[1] == qelm.n_obs * qelm.n_multiplex
    assert qrc.features.shape[1] == qelm.features.shape[1]
    assert_bundles_matched(qrc, qelm)


def test_assert_matched_passes_and_names_independent_axis_failures():
    qrc_cfg, qelm_cfg = _matched_configs()
    assert_matched(qrc_cfg, qelm_cfg)

    cases = [
        (replace(qelm_cfg, n_qubits=3), "n_qubits"),
        (replace(qelm_cfg, J_s=0.8), "J_s"),
        (replace(qelm_cfg, h=0.5), "h"),
        (replace(qelm_cfg, gamma=0.3), "gamma"),
        (replace(qelm_cfg, dissipator="dephasing"), "dissipator"),
        (replace(qelm_cfg, quantize_levels=13), "quantize_levels"),
    ]
    for bad_qelm, axis in cases:
        with pytest.raises(MatchingError, match=axis):
            assert_matched(qrc_cfg, bad_qelm)


def test_config_readout_dimension_cofires_with_multiplexing():
    qrc_cfg, qelm_cfg = _matched_configs()

    with pytest.raises(MatchingError) as excinfo:
        assert_matched(qrc_cfg, replace(qelm_cfg, n_snapshots=3))

    message = str(excinfo.value)
    assert "multiplexing" in message
    assert "readout_dimension" in message


def test_assert_matched_requires_single_shared_observable_sequence():
    qrc_cfg, qelm_cfg = _matched_configs()
    obs_qrc = readout.pauli_observables(2, axes=("x",), max_weight=1)
    obs_qelm = readout.pauli_observables(2, axes=("x", "y", "z"), max_weight=2)

    with pytest.raises(MatchingError, match="single shared"):
        assert_matched(qrc_cfg, qelm_cfg, observables=(obs_qrc, obs_qelm))


def test_assert_bundles_matched_catches_hollow_dimension_parity():
    qrc_cfg, qelm_cfg = _matched_configs()
    inputs = np.linspace(0.05, 0.95, 24)
    seed = 29
    qrc_obs = readout.pauli_observables(2, axes=("x",), max_weight=1)
    qelm_obs = readout.pauli_observables(2, axes=("x", "y", "z"), max_weight=2)

    assert_matched(qrc_cfg, qelm_cfg, observables=qrc_obs)
    qrc = build_features(qrc_cfg, inputs, seed, observables=qrc_obs)
    qelm = build_features(qelm_cfg, inputs, seed, observables=qelm_obs)

    assert qrc.features.shape[1] == 4
    assert qelm.features.shape[1] == 18
    with pytest.raises(MatchingError, match="readout_dimension"):
        assert_bundles_matched(qrc, qelm)


def test_builders_are_deterministic_for_identical_args():
    qrc_cfg, qelm_cfg = _matched_configs()
    inputs = np.linspace(0.02, 0.98, 20)
    seed = 31

    qrc_1 = build_features(qrc_cfg, inputs, seed)
    qrc_2 = build_features(qrc_cfg, inputs, seed)
    qelm_1 = build_features(qelm_cfg, inputs, seed)
    qelm_2 = build_features(qelm_cfg, inputs, seed)

    assert np.array_equal(qrc_1.features, qrc_2.features)
    assert np.array_equal(qrc_1.couplings, qrc_2.couplings)
    assert np.array_equal(qelm_1.features, qelm_2.features)
    assert np.array_equal(qelm_1.couplings, qelm_2.couplings)


def test_quantize_inputs_grid_idempotent_and_passthrough_copy():
    u = np.array([0.0, 0.123, 0.5, 0.876, 1.0])
    q = quantize_inputs(u, 201)

    assert np.array_equal(q, quantize_inputs(q, 201))
    assert np.allclose(q * 200, np.round(q * 200))

    passthrough = quantize_inputs(u, None)
    assert np.array_equal(passthrough, u)
    assert passthrough is not u
    assert np.array_equal(quantize_inputs(u, 0), u)


def test_qrc_builder_returns_full_length_rows():
    qrc_cfg, _ = _matched_configs()
    inputs = np.linspace(0.05, 0.95, 17)

    bundle = build_qrc_features(inputs, qrc_cfg, coupling_seed=5)

    assert bundle.features.shape[0] == len(inputs)
    assert bundle.first_valid == 0


def _qrc_v1_liveness_config(*, h_in: float) -> QRCConfig:
    return QRCConfig(
        dt=0.5,
        h_in=h_in,
        n_virtual=1,
        n_qubits=4,
        J_s=0.7,
        h=0.4,
        gamma=0.2,
        dissipator="local_loss",
    )


def test_qrc_v1_live_features_skip_multiplex_gain_gate():
    inputs = np.random.default_rng(20240213).uniform(0.02, 0.98, 48)
    bundle = build_qrc_features(
        inputs,
        _qrc_v1_liveness_config(h_in=1.0),
        coupling_seed=19,
    )

    diag = effective_dimension(bundle.features[bundle.first_valid :])
    assert bundle.model == "qrc"
    assert bundle.n_multiplex == 1
    assert diag["n_dead_cols"] == 0
    assert_features_live(bundle)


def test_qrc_without_input_drive_raises_dead_column_floor():
    inputs = np.random.default_rng(20240213).uniform(0.02, 0.98, 48)
    bundle = build_qrc_features(
        inputs,
        _qrc_v1_liveness_config(h_in=0.0),
        coupling_seed=19,
    )

    diag = effective_dimension(bundle.features[bundle.first_valid :])
    assert bundle.model == "qrc"
    assert bundle.n_multiplex == 1
    assert diag["n_dead_cols"] > 0
    with pytest.raises(MatchingError, match="dead_features"):
        assert_features_live(bundle)


def test_recommend_T_evolve_window_is_strict_for_one_snapshot():
    base = QELMConfig(
        d=2,
        T_evolve=0.1,
        n_snapshots=1,
        n_qubits=2,
        J_s=0.7,
        h=0.4,
        gamma=0.2,
        dissipator="local_loss",
    )

    rec = recommend_T_evolve(base, coupling_seed=7)

    assert rec["t_lo"] < rec["t_recommended"] < rec["t_hi"]


def test_effective_dimension_dead_column_threshold_is_configurable():
    features = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.001],
            [4.0, 0.002],
        ]
    )

    assert effective_dimension(features, min_col_var=1e-8)["n_dead_cols"] == 0
    assert effective_dimension(features, min_col_var=1e-6)["n_dead_cols"] == 1


def test_effective_dimension_liveness_guard_and_recommendations():
    seed = 7
    inputs = np.random.default_rng(123).uniform(0.02, 0.98, 48)
    base = QELMConfig(
        d=4,
        T_evolve=0.1,
        n_snapshots=4,
        n_qubits=4,
        J_s=0.7,
        h=0.4,
        gamma=0.3,
        dissipator="local_loss",
    )

    rec = recommend_T_evolve(base, seed)
    assert rec["regime"] == "dissipative"
    assert np.isfinite([rec["t_lo"], rec["t_recommended"], rec["t_hi"]]).all()
    assert rec["t_lo"] < rec["t_recommended"] < rec["t_hi"]

    unitary = replace(base, gamma=0.0)
    unitary_rec = recommend_T_evolve(unitary, seed)
    assert unitary_rec["regime"] == "unitary"
    assert np.isfinite(
        [unitary_rec["t_lo"], unitary_rec["t_recommended"], unitary_rec["t_hi"]]
    ).all()

    dead = build_features(replace(base, T_evolve=0.0), inputs, seed)
    dead_gain = multiplex_efficacy(
        dead.features[dead.first_valid :],
        dead.n_multiplex,
        dead.n_obs,
    )
    assert dead_gain["gain"] == pytest.approx(1.0)
    with pytest.raises(MatchingError, match="multiplex_gain"):
        assert_features_live(dead)

    large = build_features(
        replace(base, T_evolve=100.0 * rec["timescale"]),
        inputs,
        seed,
    )
    with pytest.raises(MatchingError, match="dead_features"):
        assert_features_live(large)

    live = build_features(
        replace(base, T_evolve=rec["t_recommended"]),
        inputs,
        seed,
    )
    assert_features_live(live)
    diag = effective_dimension(live.features[live.first_valid :])
    gain = multiplex_efficacy(
        live.features[live.first_valid :],
        live.n_multiplex,
        live.n_obs,
    )
    assert diag["n_dead_cols"] == 0
    assert diag["eff_rank_frac"] < 0.5
    assert gain["gain"] >= 1.15


def test_v1_live_features_skip_multiplex_gain_gate():
    seed = 7
    inputs = np.random.default_rng(123).uniform(0.02, 0.98, 48)
    base = QELMConfig(
        d=4,
        T_evolve=0.1,
        n_snapshots=1,
        n_qubits=4,
        J_s=0.7,
        h=0.4,
        gamma=0.3,
        dissipator="local_loss",
    )
    rec = recommend_T_evolve(base, seed)
    live = build_features(
        replace(base, T_evolve=rec["t_recommended"]),
        inputs,
        seed,
    )

    assert live.n_multiplex == 1
    assert_features_live(live)


def test_all_dead_columns_raise_variance_floor():
    dead = FeatureBundle(
        features=np.ones((8, 3)),
        model="synthetic",
        config={},
        couplings=np.zeros((0,)),
        n_obs=3,
        n_multiplex=1,
        first_valid=0,
    )

    with pytest.raises(MatchingError, match="dead_features=3"):
        assert_features_live(dead)


def test_physical_shot_budget_formula():
    assert physical_shot_budget(100, 12, 18) == 100 * 12 * 18
