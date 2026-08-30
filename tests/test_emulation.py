import numpy as np
import pytest

from qrc_vs_qelm.emulation import (
    emulation_metrics,
    fit_multivariate_ridge,
    paired_window_certificate,
    row_rms,
    sample_feature_matrix,
    suffix_diameter,
    tail_suffix_certificate,
    window_lower_bound,
)


def test_suffix_diameter_and_half_diameter_lower_bound():
    first = np.array([[1.0, 0.0], [0.0, 0.0]])
    second = np.array([[-1.0, 0.0], [0.0, 1.0]])

    expected_distances = np.array([np.sqrt(2.0), np.sqrt(0.5)])
    assert np.allclose(row_rms(first, second), expected_distances)
    assert suffix_diameter(first, second) == pytest.approx(np.sqrt(2.0))
    assert window_lower_bound(first, second) == pytest.approx(1.0 / np.sqrt(2.0))


def test_paired_certificate_is_tight_at_pair_midpoint():
    first = np.array([[1.0, -1.0], [0.5, 0.5]])
    second = np.array([[-1.0, 1.0], [-0.5, -0.5]])
    midpoint = 0.5 * (first + second)

    certificate = paired_window_certificate(first, second, midpoint)

    assert certificate.bound_satisfied
    assert certificate.prediction_disagreement == 0.0
    assert certificate.worst_arm_error == pytest.approx(certificate.lower_bound)
    assert certificate.paired_rmse == pytest.approx(
        certificate.paired_rmse_lower_bound
    )
    assert certificate.bound_slack == pytest.approx(0.0)
    assert certificate.paired_rmse_slack == pytest.approx(0.0)


def test_paired_certificate_rejects_non_window_predictions():
    first = np.array([[0.8, 0.2]])
    second = np.array([[-0.8, -0.2]])

    certificate = paired_window_certificate(first, second, first, second)

    assert not certificate.bound_satisfied
    assert certificate.prediction_disagreement > 0.0


def test_tail_suffix_witness_uses_tail_max_and_is_monotone():
    endpoints = (1, 2, 4)
    # Shape: endpoints x pairs x outputs.  The largest separation is at L=2.
    first = np.array([[[0.5]], [[1.0]], [[0.2]]])
    second = -first
    predictions = np.zeros_like(first)

    full = tail_suffix_certificate(endpoints, first, second, predictions)
    tail = tail_suffix_certificate(
        endpoints[1:], first[1:], second[1:], predictions[1:]
    )
    last = tail_suffix_certificate(
        endpoints[2:], first[2:], second[2:], predictions[2:]
    )

    assert full.bound_satisfied
    assert full.rms_witness_endpoint == 2
    assert full.suffix_diameter == pytest.approx(2.0)
    assert full.paired_rmse_at_witness == pytest.approx(
        full.paired_rmse_lower_bound
    )
    assert full.suffix_diameter >= tail.suffix_diameter >= last.suffix_diameter
    assert (
        full.rms_suffix_separation
        >= tail.rms_suffix_separation
        >= last.rms_suffix_separation
    )


def test_multivariate_ridge_recovers_linear_teacher_and_is_deterministic():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(90, 4))
    weights = np.array(
        [
            [1.0, -0.5, 0.2],
            [0.3, 0.1, 0.7],
            [-0.4, 0.8, 0.0],
            [0.6, -0.2, 0.5],
        ]
    )
    targets = features @ weights + np.array([0.2, -0.1, 0.4])
    train = np.arange(0, 50)
    val = np.arange(50, 70)
    test = np.arange(70, 90)

    first = fit_multivariate_ridge(
        features,
        targets,
        train,
        val,
        alphas=(0.0, 1e-8, 1e-4),
    )
    second = fit_multivariate_ridge(
        features,
        targets,
        train,
        val,
        alphas=(0.0, 1e-8, 1e-4),
    )

    assert first.alpha == 0.0
    assert np.array_equal(first.weights, second.weights)
    assert np.allclose(first.predict(features[test]), targets[test], atol=1e-11)
    assert emulation_metrics(targets[test], first.predict(features[test])).nrmse < 1e-10


def test_zero_ridge_discards_numerically_null_feature_directions():
    rng = np.random.default_rng(191)
    base = rng.normal(size=80)
    microscopic = 1e-13 * rng.normal(size=80)
    features = np.column_stack([base, base + microscopic])
    targets = rng.normal(size=(80, 2))
    train = np.arange(0, 45)
    val = np.arange(45, 60)

    model = fit_multivariate_ridge(
        features,
        targets,
        train,
        val,
        alphas=(0.0,),
    )
    perturbed = features.copy()
    perturbed[:, 1] += 1e-15

    assert np.linalg.norm(model.weights) < 10.0
    assert np.max(np.abs(model.predict(features) - model.predict(perturbed))) < 1e-12

    tiny_ridge = fit_multivariate_ridge(
        features,
        targets,
        train,
        val,
        alphas=(1e-10,),
    )
    assert np.linalg.norm(tiny_ridge.weights) < 10.0
    assert (
        np.max(
            np.abs(
                tiny_ridge.predict(features) - tiny_ridge.predict(perturbed)
            )
        )
        < 1e-12
    )


def test_multivariate_ridge_standardizer_uses_training_rows_only():
    features = np.arange(40.0).reshape(20, 2)
    targets = np.column_stack([features[:, 0], features[:, 1]])
    train = np.arange(0, 8)
    val = np.arange(8, 12)
    shifted = features.copy()
    shifted[12:] += 1e6

    model = fit_multivariate_ridge(
        shifted,
        targets,
        train,
        val,
        alphas=(0.0,),
    )

    assert np.array_equal(model.feature_mean, features[train].mean(axis=0))
    assert np.array_equal(model.feature_scale, features[train].std(axis=0))


def test_positive_infinite_ridge_is_exact_intercept_only_candidate():
    rng = np.random.default_rng(71)
    features = rng.normal(size=(12, 3))
    targets = rng.normal(size=(12, 2))
    train = np.arange(0, 6)
    val = np.arange(6, 9)

    model = fit_multivariate_ridge(
        features,
        targets,
        train,
        val,
        alphas=(float("inf"),),
    )

    expected = targets[np.concatenate([train, val])].mean(axis=0)
    assert np.isposinf(model.alpha)
    assert np.array_equal(model.weights, np.zeros_like(model.weights))
    assert np.allclose(model.predict(features), expected[None, :])


def test_sample_feature_matrix_exact_copy_and_seeded_shots():
    features = np.array([[0.1, -0.2], [0.3, 0.4]])
    exact = sample_feature_matrix(features, None, rng=np.random.default_rng(1))
    first = sample_feature_matrix(features, 1000, rng=np.random.default_rng(2))
    second = sample_feature_matrix(features, 1000, rng=np.random.default_rng(2))

    assert np.array_equal(exact, features)
    assert exact is not features
    assert np.array_equal(first, second)
    assert np.all(first >= -1.0)
    assert np.all(first <= 1.0)


def test_emulation_validation_rejects_shape_and_split_errors():
    with pytest.raises(ValueError, match="identical shapes"):
        row_rms(np.ones((2, 2)), np.ones((2, 3)))

    features = np.ones((6, 2))
    targets = np.ones((6, 1))
    with pytest.raises(ValueError, match="disjoint"):
        fit_multivariate_ridge(
            features,
            targets,
            np.array([0, 1, 2]),
            np.array([2, 3]),
            alphas=(0.0,),
        )
