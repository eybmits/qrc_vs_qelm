import numpy as np
import pytest

from qrc_vs_qelm import controls, ipc, pipeline, task_families
from qrc_vs_qelm.core import readout
from qrc_vs_qelm.models.api import QRCConfig, build_qrc_features


def _splits_for(u, basis, max_delay):
    targets = {spec.key: ipc.basis_target(u, spec) for spec in basis}
    idx = pipeline.valid_indices(len(u), washout=0, max_delay=max_delay, targets=targets)
    return pipeline.make_splits(idx)


def test_basis_enumeration_order_truncation_and_product_target(caplog):
    full = ipc.dambre_basis(
        max_delay=2,
        max_degree=2,
        max_total_degree=3,
        max_functions=20,
    )
    repeated = ipc.dambre_basis(
        max_delay=2,
        max_degree=2,
        max_total_degree=3,
        max_functions=20,
    )

    assert [spec.key for spec in full] == [spec.key for spec in repeated]
    assert [spec.key for spec in full[:9]] == [
        "ipc_d1_tau0_p1",
        "ipc_d1_tau1_p1",
        "ipc_d1_tau2_p1",
        "ipc_d2_tau0_p2",
        "ipc_d2_tau1_p2",
        "ipc_d2_tau2_p2",
        "ipc_d2_tau0_p1_tau1_p1",
        "ipc_d2_tau0_p1_tau2_p1",
        "ipc_d2_tau1_p1_tau2_p1",
    ]
    assert not full.truncated

    capped = ipc.dambre_basis(2, 2, 3, max_functions=5)
    assert len(capped) == 5
    assert capped.truncated
    assert "truncated" in caplog.text

    u = np.linspace(0.05, 0.95, 12)
    spec = ipc.BasisSpec(
        factors=((0, 1), (2, 2)),
        total_degree=3,
        key="manual",
    )
    got = ipc.basis_target(u, spec)
    expected = task_families.legendre_target(
        u,
        tau=0,
        degree=1,
    ) * task_families.legendre_target(u, tau=2, degree=2)
    assert np.isnan(got[:2]).all()
    assert np.allclose(got[2:], expected[2:])

    with pytest.raises(ValueError, match="max_delay"):
        ipc.dambre_basis(-1, 2, 3, max_functions=20)
    with pytest.raises(ValueError, match="max_degree"):
        ipc.dambre_basis(2, 0, 3, max_functions=20)
    with pytest.raises(ValueError, match="max_total_degree"):
        ipc.dambre_basis(2, 2, 0, max_functions=20)
    with pytest.raises(TypeError, match="BasisSpec"):
        ipc.basis_target(u, {"factors": ((0, 1),), "total_degree": 1, "key": "x"})


def test_total_ipc_bound_on_synthetic_full_rank_and_small_qrc():
    rng = np.random.default_rng(112)
    T = 220
    u = rng.uniform(0.0, 1.0, size=T)
    basis = ipc.dambre_basis(
        max_delay=2,
        max_degree=1,
        max_total_degree=1,
        max_functions=3,
    )
    splits = _splits_for(u, basis, max_delay=2)
    X = controls.linear_delay_features(u, d=3, D=3)

    result = ipc.total_ipc(
        X,
        u,
        basis,
        splits,
        alphas=(0.0,),
        rng=np.random.default_rng(1),
    )
    rank = np.linalg.matrix_rank(X[splits.all] - X[splits.all].mean(axis=0))
    assert result["total_ipc"] <= rank + 1e-9
    assert result["effective_rank"] == rank
    assert result["raw_sum"] >= result["total_ipc"]
    assert rank <= X.shape[1]
    assert result["total_ipc"] == pytest.approx(3.0, abs=1e-9)

    qrc_u = np.random.default_rng(321).uniform(0.02, 0.98, size=120)
    observables = readout.pauli_observables(2, axes=("x", "z"), max_weight=1)
    cfg = QRCConfig(
        dt=0.25,
        h_in=0.7,
        n_virtual=1,
        n_qubits=2,
        J_s=0.55,
        h=0.25,
        gamma=0.1,
        dissipator="local_loss",
        quantize_levels=17,
    )
    bundle = build_qrc_features(qrc_u, cfg, coupling_seed=11, observables=observables)
    qrc_basis = ipc.dambre_basis(2, 2, 2, max_functions=9)
    qrc_targets = {spec.key: ipc.basis_target(qrc_u, spec) for spec in qrc_basis}
    qrc_idx = pipeline.stage_valid_indices(
        [bundle],
        qrc_targets,
        washout=1,
        max_delay=2,
    )
    qrc_splits = pipeline.make_splits(qrc_idx)
    qrc_result = ipc.total_ipc(
        bundle.features,
        qrc_u,
        qrc_basis,
        qrc_splits,
        alphas=(1e-6, 1e-3, 1e-1),
        rng=np.random.default_rng(2),
    )
    qrc_rows = bundle.features[qrc_splits.all]
    qrc_rank = np.linalg.matrix_rank(qrc_rows - qrc_rows.mean(axis=0))
    assert qrc_result["total_ipc"] <= qrc_rank + 1e-9
    assert qrc_result["by_delay"]
    assert qrc_rank <= bundle.features.shape[1]


def test_linear_delay_line_ipc_is_memory_capacity_not_nonlinear_capacity():
    rng = np.random.default_rng(123)
    T = 400
    u = rng.uniform(0.0, 1.0, size=T)
    basis = ipc.dambre_basis(
        max_delay=3,
        max_degree=2,
        max_total_degree=2,
        max_functions=20,
    )
    splits = _splits_for(u, basis, max_delay=3)
    X = controls.linear_delay_features(u, d=4, D=4)

    result = ipc.total_ipc(
        X,
        u,
        basis,
        splits,
        alphas=(0.0, 1e-8),
        threshold=0.05,
        rng=np.random.default_rng(4),
    )

    assert result["mc"] == pytest.approx(4.0, abs=1e-9)
    assert result["nc"] == pytest.approx(0.0, abs=1e-12)
    assert result["total_ipc"] <= np.linalg.matrix_rank(X[splits.all]) + 1e-9


def test_surrogate_significance_cut_removes_noise_feature_capacity():
    rng = np.random.default_rng(456)
    T = 420
    u = rng.uniform(0.0, 1.0, size=T)
    X = rng.normal(size=(T, 5))
    basis = ipc.dambre_basis(
        max_delay=3,
        max_degree=2,
        max_total_degree=2,
        max_functions=14,
    )
    splits = _splits_for(u, basis, max_delay=3)

    result = ipc.total_ipc(
        X,
        u,
        basis,
        splits,
        alphas=(1e-6, 1e-3),
        n_surrogates=24,
        rng=np.random.default_rng(789),
    )

    assert result["threshold"] > 0.0
    assert result["thresholds_by_degree"]
    assert (
        result["surrogate_retained_capacity_max"]
        <= ipc.DEFAULT_CUMULATIVE_SURROGATE_EPSILON + 1e-12
    )
    assert result["total_ipc"] == pytest.approx(0.0, abs=1e-12)
    assert result["n_significant"] == 0


def test_default_surrogate_path_zeroes_bilinear_leak_and_clamps_bound_over_seeds():
    basis = ipc.dambre_basis(
        max_delay=3,
        max_degree=2,
        max_total_degree=3,
        max_functions=25,
    )
    bilinear_key = "ipc_d2_tau0_p1_tau1_p1"
    assert bilinear_key in {spec.key for spec in basis}

    for seed in range(20):
        rng = np.random.default_rng(10_000 + seed)
        T = 180 + 2 * (seed % 3)
        u = rng.uniform(0.0, 1.0, size=T)
        X = controls.linear_delay_features(u, d=4, D=4)
        splits = _splits_for(u, basis, max_delay=3)

        result = ipc.total_ipc(
            X,
            u,
            basis,
            splits,
            alphas=(0.0, 1e-8, 1e-4),
            rng=np.random.default_rng(20_000 + seed),
        )
        rows = X[splits.all]
        rank = np.linalg.matrix_rank(rows - rows.mean(axis=0))

        assert result["retained_capacities_by_key"][bilinear_key] == 0.0
        assert result["raw_by_degree"].get(2, 0.0) == pytest.approx(0.0, abs=1e-12)
        assert result["total_ipc"] <= rank + 1e-9
        assert result["effective_rank"] == rank


def test_default_surrogate_path_enforces_bound_on_small_qrc_features():
    basis = ipc.dambre_basis(
        max_delay=3,
        max_degree=2,
        max_total_degree=3,
        max_functions=25,
    )
    observables = readout.pauli_observables(2, axes=("x", "z"), max_weight=1)
    cfg = QRCConfig(
        dt=0.25,
        h_in=0.7,
        n_virtual=1,
        n_qubits=2,
        J_s=0.55,
        h=0.25,
        gamma=0.1,
        dissipator="local_loss",
        quantize_levels=17,
    )

    for seed in range(3):
        rng = np.random.default_rng(30_000 + seed)
        u = rng.uniform(0.02, 0.98, size=150 + 20 * seed)
        bundle = build_qrc_features(u, cfg, coupling_seed=100 + seed, observables=observables)
        targets = {spec.key: ipc.basis_target(u, spec) for spec in basis}
        idx = pipeline.stage_valid_indices(
            [bundle],
            targets,
            washout=1,
            max_delay=3,
        )
        splits = pipeline.make_splits(idx)

        result = ipc.total_ipc(
            bundle.features,
            u,
            basis,
            splits,
            alphas=(1e-6, 1e-3, 1e-1),
            rng=np.random.default_rng(40_000 + seed),
        )
        rows = bundle.features[splits.all]
        rank = np.linalg.matrix_rank(rows - rows.mean(axis=0))

        assert result["total_ipc"] <= rank + 1e-9
        assert result["effective_rank"] == rank
        assert result["by_delay"]


def test_production_basis_default_surrogate_path_holds_theorem_bound():
    basis = ipc.dambre_basis(
        max_delay=20,
        max_degree=4,
        max_total_degree=4,
        max_functions=600,
    )
    assert len(basis) == 600
    assert basis.truncated

    for T, seed in ((200, 50_001), (600, 50_002)):
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, 1.0, size=T)
        X = controls.linear_delay_features(u, d=8, D=8)
        splits = _splits_for(u, basis, max_delay=20)

        result = ipc.total_ipc(
            X,
            u,
            basis,
            splits,
            alphas=(0.0, 1e-8, 1e-4),
            rng=np.random.default_rng(seed + 100),
        )
        rows = X[splits.all]
        rank = np.linalg.matrix_rank(rows - rows.mean(axis=0))

        assert result["total_ipc"] <= rank + 1e-9
        assert result["effective_rank"] == rank
        assert result["bound_excess"] <= 1e-9
        assert result["raw_sum"] <= rank + 1e-9
        assert set(result["by_delay"]).issubset(set(range(21)))


def test_total_ipc_is_deterministic_for_same_rng_seed():
    rng = np.random.default_rng(999)
    T = 180
    u = rng.uniform(0.0, 1.0, size=T)
    X = controls.linear_delay_features(u, d=3, D=5)
    basis = ipc.dambre_basis(
        max_delay=2,
        max_degree=2,
        max_total_degree=2,
        max_functions=9,
    )
    splits = _splits_for(u, basis, max_delay=2)

    first = ipc.total_ipc(
        X,
        u,
        basis,
        splits,
        alphas=(1e-8, 1e-4),
        n_surrogates=8,
        rng=np.random.default_rng(55),
    )
    second = ipc.total_ipc(
        X,
        u,
        basis,
        splits,
        alphas=(1e-8, 1e-4),
        n_surrogates=8,
        rng=np.random.default_rng(55),
    )

    assert first == second
