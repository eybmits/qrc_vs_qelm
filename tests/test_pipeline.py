import numpy as np
import pytest

from qrc_vs_qelm import controls, pipeline, task_families
from qrc_vs_qelm.core import readout
from qrc_vs_qelm.matching import (
    assert_bundles_matched,
    assert_features_live,
    participation_effective_rank,
)
from qrc_vs_qelm.models.api import FeatureBundle, QELMConfig, QRCConfig, build_features


def _old_bias_penalized_predict(X_train, y_train, X_eval, alpha):
    weights = readout.train_readout(
        readout.add_bias(X_train),
        y_train,
        ridge=float(alpha),
    )
    return readout.predict(readout.add_bias(X_eval), weights)


def test_valid_indices_splits_and_train_only_standardization():
    T = 12
    y = np.arange(T, dtype=float)
    y[:3] = np.nan
    y[8] = np.nan

    idx = pipeline.valid_indices(T, washout=2, max_delay=4, targets={"y": y})
    assert np.array_equal(idx, np.array([4, 5, 6, 7, 9, 10, 11]))

    splits = pipeline.make_splits(idx)
    joined = splits.all
    assert np.array_equal(joined, idx)
    assert np.intersect1d(splits.train, splits.val).size == 0
    assert np.intersect1d(splits.train, splits.test).size == 0
    assert np.intersect1d(splits.val, splits.test).size == 0
    assert np.all(np.diff(joined) > 0)

    X = np.column_stack([np.arange(T, dtype=float), np.ones(T)])
    changed = X.copy()
    changed[splits.test, 0] += 10_000.0

    Z = pipeline.standardize_train_only(X, splits)
    Z_changed = pipeline.standardize_train_only(changed, splits)
    assert np.array_equal(Z[splits.train], Z_changed[splits.train])
    assert np.allclose(Z[splits.train, 1], 0.0)


def test_make_splits_boundary_min_sizes_and_near_degenerate_fractions():
    idx = np.array([10, 11, 12])

    splits = pipeline.make_splits(idx, fractions=(0.98, 0.01, 0.01))
    assert splits.train.size == 1
    assert splits.val.size == 1
    assert splits.test.size == 1
    assert np.array_equal(splits.all, idx)

    near = pipeline.make_splits(np.arange(11), fractions=(1e-12, 1e-12, 1.0))
    assert near.train.size == 1
    assert near.val.size == 1
    assert near.test.size == 9
    assert np.array_equal(near.all, np.arange(11))


def test_evaluate_feature_matrix_columns_context_and_nan_guard():
    rng = np.random.default_rng(5)
    T = 60
    X = rng.normal(size=(T, 5))
    targets = task_families.build_family_targets(
        rng.uniform(0.0, 1.0, size=T),
        taus=[2, 4],
        degrees=[1],
    )
    idx = pipeline.valid_indices(T, washout=5, max_delay=4, targets=targets)
    splits = pipeline.make_splits(idx)

    df = pipeline.evaluate_feature_matrix(X, targets, splits)

    assert list(df.columns) == [
        "target",
        "alpha",
        "capacity_test",
        "nrmse_test",
        "n_train",
        "n_val",
        "n_test",
        "effective_rank",
        "feature_dim",
    ]
    assert set(df["target"]) == set(targets)
    assert set(df["feature_dim"]) == {5}
    assert set(df["n_train"]) == {splits.train.size}
    assert set(df["n_val"]) == {splits.val.size}
    assert set(df["n_test"]) == {splits.test.size}
    Xs = pipeline.standardize_train_only(X, splits)
    expected_rank = participation_effective_rank(Xs[splits.train])
    assert np.allclose(df["effective_rank"], expected_rank)

    ctx = df.attrs["evaluation_context"]
    assert np.array_equal(ctx.indices, splits.all)
    for name, target in targets.items():
        assert np.array_equal(ctx.targets[name], target[ctx.indices])

    bad_splits = pipeline.make_splits(pipeline.valid_indices(T, 0, 0))
    with pytest.raises(ValueError, match="non-finite"):
        pipeline.evaluate_feature_matrix(X, targets, bad_splits)

    X_bad = X.copy()
    X_bad[splits.val[0], 3] = np.inf
    with pytest.raises(ValueError, match=r"val split at row .* column 3"):
        pipeline.evaluate_feature_matrix(X_bad, targets, splits)


def test_unpenalized_intercept_bounds_high_alpha_nonzero_mean_nrmse():
    rng = np.random.default_rng(41)
    T = 90
    u = rng.uniform(0.01, 0.99, size=T)
    X = (2.0 * u - 1.0).reshape(-1, 1)
    y = task_families.legendre_target(u, tau=0, degree=1) + 5.0
    targets = {"offset_legendre": y}
    splits = pipeline.make_splits(pipeline.valid_indices(T, 0, 0, targets))

    df = pipeline.evaluate_feature_matrix(X, targets, splits)
    assert float(df.loc[0, "nrmse_test"]) < 1e-5

    high_alpha = pipeline.evaluate_feature_matrix(
        X,
        targets,
        splits,
        alphas=(1e2,),
    )
    assert np.isfinite(high_alpha.loc[0, "nrmse_test"])
    assert float(high_alpha.loc[0, "nrmse_test"]) < 1.0


def test_centered_ridge_does_not_use_eval_rows_for_intercept_fit():
    X_train = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y_train = np.array([3.0, 4.0, 5.0, 6.0])
    X_eval = np.array([[0.5], [10.0]])
    X_eval_changed = np.array([[0.5], [1e9]])

    pred = pipeline._fit_predict(X_train, y_train, X_eval, alpha=1e2)
    changed = pipeline._fit_predict(
        X_train,
        y_train,
        X_eval_changed,
        alpha=1e2,
    )

    assert pred[0] == pytest.approx(changed[0])


def test_zero_mean_capacity_matches_old_bias_penalized_path():
    T = 80
    u = np.linspace(0.0, 1.0, T)
    X = (2.0 * u - 1.0).reshape(-1, 1)
    y = X[:, 0].copy()
    targets = {"zero_mean": y}
    splits = pipeline.make_splits(pipeline.valid_indices(T, 0, 0, targets))

    df = pipeline.evaluate_feature_matrix(X, targets, splits, alphas=(1e2,))
    Xs = pipeline.standardize_train_only(X, splits)
    train_val = np.concatenate([splits.train, splits.val])
    old_pred = _old_bias_penalized_predict(
        Xs[train_val],
        y[train_val],
        Xs[splits.test],
        alpha=1e2,
    )

    assert float(df.loc[0, "capacity_test"]) == pytest.approx(
        readout.capacity(y[splits.test], old_pred),
        abs=1e-6,
    )


def test_stage_valid_indices_excludes_qelm_warmup_for_tau_zero_target():
    rng = np.random.default_rng(37)
    T = 18
    u = rng.uniform(0.02, 0.98, size=T)
    qelm_cfg = QELMConfig(
        d=3,
        T_evolve=0.4,
        n_snapshots=1,
        n_qubits=3,
        J_s=0.7,
        h=0.4,
        gamma=0.2,
        dissipator="local_loss",
        quantize_levels=17,
    )
    bundle = build_features(qelm_cfg, u, coupling_seed=11)
    targets = task_families.build_family_targets(u, taus=[0], degrees=[1])

    plain = pipeline.valid_indices(T, washout=0, max_delay=0, targets=targets)
    assert np.array_equal(plain[:2], np.array([0, 1]))

    idx = pipeline.stage_valid_indices(
        [bundle],
        targets,
        washout=0,
        max_delay=0,
    )
    assert bundle.first_valid == 2
    assert idx[0] == bundle.first_valid
    assert 0 not in idx
    assert 1 not in idx

    splits = pipeline.make_splits(idx)
    assert splits.all.min() >= bundle.first_valid
    pipeline.evaluate_feature_matrix(
        bundle.features,
        targets,
        splits,
        first_valid=bundle.first_valid,
    )

    unsafe_splits = pipeline.make_splits(plain)
    with pytest.raises(ValueError, match="first_valid=2"):
        pipeline.evaluate_feature_matrix(
            bundle.features,
            targets,
            unsafe_splits,
            first_valid=bundle.first_valid,
        )


def test_stage_valid_indices_uses_most_restrictive_first_valid_for_k_sweep():
    rng = np.random.default_rng(20260706)
    T = 36
    u = rng.uniform(0.02, 0.98, size=T)
    targets = task_families.build_family_targets(u, taus=[0, 2], degrees=[1])
    features = rng.normal(size=(T, 4))

    def bundle(model, first_valid):
        return FeatureBundle(
            features=features.copy(),
            model=model,
            config={},
            couplings=np.zeros((1, 1)),
            n_obs=4,
            n_multiplex=1,
            first_valid=first_valid,
        )

    qrc_bundle = bundle("qrc", 0)
    qelm_k2 = bundle("qelm", 1)
    qelm_k6 = bundle("qelm", 5)

    idx = pipeline.stage_valid_indices(
        [qrc_bundle, qelm_k2, qelm_k6],
        targets,
        washout=0,
        max_delay=2,
    )
    splits = pipeline.make_splits(idx)
    assert idx[0] == qelm_k6.first_valid

    qrc_frame = pipeline.evaluate_feature_matrix(
        qrc_bundle.features,
        targets,
        splits,
        first_valid=qrc_bundle.first_valid,
    )
    qelm_frame = pipeline.evaluate_feature_matrix(
        qelm_k2.features,
        targets,
        splits,
        first_valid=qelm_k2.first_valid,
    )
    qrc_ctx = qrc_frame.attrs["evaluation_context"]
    qelm_ctx = qelm_frame.attrs["evaluation_context"]

    assert np.array_equal(qrc_ctx.indices, qelm_ctx.indices)
    assert np.array_equal(qrc_ctx.train_idx, qelm_ctx.train_idx)
    assert np.array_equal(qrc_ctx.val_idx, qelm_ctx.val_idx)
    assert np.array_equal(qrc_ctx.test_idx, qelm_ctx.test_idx)


def test_matched_qrc_qelm_controls_share_targets_and_indices():
    rng = np.random.default_rng(202406)
    T = 72
    u = rng.uniform(0.02, 0.98, size=T)
    seed = 19

    qrc_cfg = QRCConfig(
        dt=0.5,
        h_in=1.0,
        n_virtual=1,
        n_qubits=4,
        J_s=0.7,
        h=0.4,
        gamma=0.2,
        dissipator="local_loss",
        quantize_levels=17,
    )
    qelm_cfg = QELMConfig(
        d=4,
        T_evolve=0.8,
        n_snapshots=1,
        n_qubits=4,
        J_s=0.7,
        h=0.4,
        gamma=0.2,
        dissipator="local_loss",
        quantize_levels=17,
    )

    qrc_bundle = build_features(qrc_cfg, u, seed)
    qelm_bundle = build_features(qelm_cfg, u, seed)
    assert qrc_bundle.n_multiplex == 1
    assert qelm_bundle.n_multiplex == 1
    assert_bundles_matched(qrc_bundle, qelm_bundle)
    assert_features_live(qrc_bundle)
    assert_features_live(qelm_bundle)

    D = qrc_bundle.features.shape[1]
    matrices = [
        qrc_bundle.features,
        qelm_bundle.features,
        controls.linear_delay_features(u, d=qelm_cfg.d, D=D),
        controls.poly_delay_features(u, d=qelm_cfg.d, D=D),
    ]
    assert all(X.shape == (T, D) for X in matrices)

    targets = task_families.build_family_targets(u, taus=[0, 2, 5], degrees=[1])
    idx = pipeline.stage_valid_indices(
        [qrc_bundle, qelm_bundle],
        targets,
        washout=1,
        max_delay=5,
    )
    splits = pipeline.make_splits(idx)
    frames = [
        pipeline.evaluate_feature_matrix(X, targets, splits, first_valid=idx[0])
        for X in matrices
    ]

    base_ctx = frames[0].attrs["evaluation_context"]
    for df in frames[1:]:
        ctx = df.attrs["evaluation_context"]
        assert np.array_equal(ctx.indices, base_ctx.indices)
        assert np.array_equal(ctx.train_idx, base_ctx.train_idx)
        assert np.array_equal(ctx.val_idx, base_ctx.val_idx)
        assert np.array_equal(ctx.test_idx, base_ctx.test_idx)
        for name in targets:
            assert np.array_equal(ctx.targets[name], base_ctx.targets[name])
