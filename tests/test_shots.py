import numpy as np

from qrc_vs_qelm.shots import add_shot_noise


def test_constant_feature_column_empirical_std_matches_formula():
    f = 0.2
    shots = 400
    features = np.full((20_000, 1), f)
    noisy = add_shot_noise(features, shots, np.random.default_rng(123))

    empirical = float(np.std(noisy[:, 0], ddof=1))
    expected = float(np.sqrt((1.0 - f**2) / shots))
    assert abs(empirical - expected) / expected < 0.05


def test_shot_noise_clips_outputs_and_is_rng_deterministic():
    features = np.array([[-0.9, 0.0, 0.9], [0.3, -0.2, 0.1]])
    noisy_1 = add_shot_noise(features, 25, np.random.default_rng(9))
    noisy_2 = add_shot_noise(features, 25, np.random.default_rng(9))

    assert np.array_equal(noisy_1, noisy_2)
    assert np.all(noisy_1 >= -1.0)
    assert np.all(noisy_1 <= 1.0)


def test_infinite_shots_return_exact_float64_copy():
    features = np.array([[1, 2], [3, 4]], dtype=np.int64)
    out = add_shot_noise(features, np.inf, np.random.default_rng(1))

    assert out.dtype == np.float64
    assert np.array_equal(out, features.astype(np.float64))
    assert out is not features


def test_features_are_clipped_before_finite_shot_noise():
    features = np.array([[2.0, -2.0, 0.0]])
    out = add_shot_noise(features, 100, np.random.default_rng(2))

    assert out[0, 0] == 1.0
    assert out[0, 1] == -1.0
    assert -1.0 <= out[0, 2] <= 1.0
