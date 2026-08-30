import numpy as np
import pytest

from qrc_vs_qelm.canonical_emulation import (
    classify_emulation_regime,
    degree_two_delay_features,
    fit_exponential_forgetting,
    fit_random_tanh_feature_map,
    nested_snapshot_columns,
    select_nested_snapshots,
    suffix_risk_decomposition,
    target_projection_diagnostic,
)


def test_suffix_risk_decomposition_is_exact_for_grouped_histories():
    means = np.array([[1.0, 2.0], [3.0, 5.0]])
    deviations = np.array(
        [
            [[-0.2, 0.4], [0.2, -0.4]],
            [[-0.6, 0.1], [0.6, -0.1]],
        ]
    )
    targets = means[:, None, :] + deviations
    predictions = means + np.array([[0.3, -0.1], [-0.2, 0.5]])

    result = suffix_risk_decomposition(targets, predictions)

    assert result.history_mse > 0.0
    assert result.representation_mse > 0.0
    assert result.total_mse == pytest.approx(
        result.history_mse + result.representation_mse,
        abs=1e-15,
    )
    assert result.identity_error == pytest.approx(0.0, abs=1e-15)
    assert result.total_nrmse**2 == pytest.approx(
        result.history_nrmse**2 + result.representation_nrmse**2
    )
    assert result.n_suffix_groups == 2
    assert result.branches_per_group == 2
    assert result.n_outputs == 2


def test_suffix_risk_decomposition_rejects_one_prediction_per_branch():
    targets = np.zeros((3, 2, 4))
    with pytest.raises(ValueError, match="predictions shape"):
        suffix_risk_decomposition(targets, np.zeros((6, 4)))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                history_nrmse=0.0,
                student_nrmse=0.0,
                oracle_nrmse=0.0,
                target_variance=1e-12,
            ),
            "DEGENERATE_TARGET",
        ),
        (
            dict(
                history_nrmse=0.3,
                student_nrmse=0.4,
                oracle_nrmse=0.35,
                target_variance=1.0,
            ),
            "HISTORY_IMPOSSIBLE",
        ),
        (
            dict(
                history_nrmse=0.02,
                student_nrmse=0.08,
                oracle_nrmse=0.03,
                target_variance=1.0,
            ),
            "EMULATED",
        ),
        (
            dict(
                history_nrmse=0.02,
                student_nrmse=0.4,
                oracle_nrmse=0.03,
                target_variance=1.0,
            ),
            "REPRESENTATION_LIMITED",
        ),
        (
            dict(
                history_nrmse=0.02,
                student_nrmse=0.4,
                oracle_nrmse=0.2,
                target_variance=1.0,
            ),
            "UNRESOLVED",
        ),
    ],
)
def test_classify_emulation_regime(kwargs, expected):
    assert (
        classify_emulation_regime(
            **kwargs,
            tolerance=0.1,
            min_target_variance=1e-8,
        )
        == expected
    )


def test_projection_diagnostic_detects_exact_linear_target_span():
    x = np.linspace(-1.0, 1.0, 20)[:, None]
    features = np.column_stack([x[:, 0], x[:, 0] ** 2])
    targets = np.column_stack(
        [1.0 + 2.0 * x[:, 0], -0.5 + 3.0 * x[:, 0] ** 2]
    )

    result = target_projection_diagnostic(features, targets)

    assert result.rank == 3
    assert result.residual_mse < 1e-25
    assert result.target_aligned_fraction == pytest.approx(1.0)


def test_nested_snapshot_columns_are_true_nested_measurement_budgets():
    one = nested_snapshot_columns(
        max_snapshots=4,
        selected_snapshots=1,
        n_observables=2,
    )
    two = nested_snapshot_columns(
        max_snapshots=4,
        selected_snapshots=2,
        n_observables=2,
    )
    four = nested_snapshot_columns(
        max_snapshots=4,
        selected_snapshots=4,
        n_observables=2,
    )

    assert np.array_equal(one, [6, 7])
    assert np.array_equal(two, [2, 3, 6, 7])
    assert np.array_equal(four, np.arange(8))
    assert set(one).issubset(two)
    assert set(two).issubset(four)

    matrix = np.arange(24, dtype=float).reshape(3, 8)
    assert np.array_equal(
        select_nested_snapshots(
            matrix,
            max_snapshots=4,
            selected_snapshots=2,
            n_observables=2,
        ),
        matrix[:, two],
    )


def test_nested_snapshot_columns_reject_nondividing_budget():
    with pytest.raises(ValueError, match="divide"):
        nested_snapshot_columns(
            max_snapshots=4,
            selected_snapshots=3,
            n_observables=2,
        )


def test_random_tanh_feature_map_is_deterministic_nested_and_train_scaled():
    train = np.array([[0.0, 2.0], [1.0, 4.0], [2.0, 6.0]])
    test = np.array([[3.0, 8.0], [4.0, 10.0]])
    first = fit_random_tanh_feature_map(train, width=8, seed=123)
    second = fit_random_tanh_feature_map(train, width=8, seed=123)

    assert np.array_equal(first.mean, np.mean(train, axis=0))
    assert np.array_equal(first.scale, np.std(train, axis=0))
    assert np.array_equal(first.transform(test), second.transform(test))
    assert np.array_equal(
        first.transform(test, width=3),
        first.transform(test)[:, :3],
    )


def test_degree_two_delay_features_have_expected_unique_terms():
    windows = np.array([[2.0, 3.0], [4.0, 5.0]])
    features = degree_two_delay_features(windows)

    assert features.shape == (2, 5)
    assert np.array_equal(features[0], [2.0, 3.0, 4.0, 6.0, 9.0])


def test_exponential_forgetting_fit_recovers_known_scale_and_window():
    windows = np.arange(0, 8, dtype=float)
    witness = 2.0 * np.exp(-windows / 3.0)

    result = fit_exponential_forgetting(windows, witness)

    assert result.amplitude == pytest.approx(2.0)
    assert result.tau == pytest.approx(3.0)
    assert result.r_squared == pytest.approx(1.0)
    assert result.predicted_window(0.1) == 9

    already_below = fit_exponential_forgetting(
        windows,
        0.05 * np.exp(-windows / 3.0),
    )
    assert already_below.predicted_window(0.1) == 1
