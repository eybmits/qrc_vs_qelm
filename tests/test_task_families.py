import numpy as np
import pytest

from qrc_vs_qelm import task_families


def test_legendre_orthogonality_and_linear_shift():
    rng = np.random.default_rng(123)
    u = rng.uniform(0.0, 1.0, size=30_000)

    targets = [task_families.legendre_target(u, tau=0, degree=p) for p in range(4)]
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            assert abs(float(np.mean(targets[i] * targets[j]))) < 1.5e-2

    shifted = task_families.legendre_target(u, tau=3, degree=1)
    assert np.isnan(shifted[:3]).all()
    assert np.array_equal(shifted[3:], 2.0 * u[:-3] - 1.0)


def test_narma2_matches_hand_rolled_ten_step_reference():
    u = np.linspace(0.05, 0.95, 12)
    got = task_families.narma_targets(u)["narma2"]

    s = 0.2 * u
    expected = np.zeros_like(u)
    for k in range(2, len(u)):
        expected[k] = (
            0.3 * expected[k - 1]
            + 0.05 * expected[k - 1] * np.sum(expected[k - 2 : k])
            + 1.5 * s[k - 2] * s[k - 1]
            + 0.1
        )
    expected[:2] = np.nan

    assert np.isnan(got[:2]).all()
    assert np.allclose(got[2:], expected[2:])


def test_parity_targets_correctness():
    bits = np.array([1, 0, 1, 1, 0, 1, 0], dtype=float)
    y = task_families.parity_targets(bits, taus=[3])["parity_tau3"]

    assert np.isnan(y[:3]).all()
    for t in range(3, len(bits)):
        assert y[t] == np.sum(bits[t - 3 : t]) % 2


def test_product_target_keys_and_santa_fe_cache_loader(tmp_path):
    u = np.linspace(0.1, 0.9, 8)
    products = task_families.build_product_targets(u, specs=[(2, 1, 2)])
    assert set(products) == {"prod_tau2_p12"}
    assert np.isnan(products["prod_tau2_p12"][:2]).all()

    missing = tmp_path / "missing.npy"
    with pytest.warns(RuntimeWarning, match="Santa Fe cache file not found"):
        assert task_families.load_santa_fe(missing) is None

    cached = tmp_path / "santa_fe.npy"
    np.save(cached, np.array([[1.0, 2.0, 3.0]]))
    assert np.array_equal(task_families.load_santa_fe(cached), np.array([1.0, 2.0, 3.0]))
