import numpy as np

from qrc_vs_qelm import controls


def test_linear_delay_features_dimension_truncation_and_zero_padding():
    u = np.array([1.0, 2.0, 3.0])

    padded = controls.linear_delay_features(u, d=3, D=5)
    assert padded.shape == (3, 5)
    assert np.array_equal(
        padded,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 0.0, 0.0],
                [3.0, 2.0, 1.0, 0.0, 0.0],
            ]
        ),
    )

    truncated = controls.linear_delay_features(u, d=3, D=2)
    assert truncated.shape == (3, 2)
    assert np.array_equal(truncated, padded[:, :2])


def test_delay_features_zero_pad_when_window_exceeds_stream_length():
    u = np.array([1.0, 2.0, 3.0])

    X = controls.linear_delay_features(u, d=5, D=5)

    assert X.shape == (3, 5)
    assert np.array_equal(
        X,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 0.0, 0.0],
                [3.0, 2.0, 1.0, 0.0, 0.0],
            ]
        ),
    )


def test_poly_delay_features_order_dimension_and_determinism():
    u = np.array([1.0, 2.0, 3.0])

    X1 = controls.poly_delay_features(u, d=3, D=10)
    X2 = controls.poly_delay_features(u, d=3, D=10)

    assert X1.shape == (3, 10)
    assert np.array_equal(X1, X2)
    assert np.array_equal(
        X1[2],
        np.array([3.0, 2.0, 1.0, 9.0, 4.0, 1.0, 6.0, 3.0, 2.0, 0.0]),
    )

    truncated = controls.poly_delay_features(u, d=3, D=4)
    assert truncated.shape == (3, 4)
    assert np.array_equal(truncated, X1[:, :4])
