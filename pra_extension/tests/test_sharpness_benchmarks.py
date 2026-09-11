"""Independent implementation checks for post-hoc analytic reference examples."""

import itertools

import numpy as np
import pytest
from scipy.stats import binom

import sharpness_benchmarks as sharp


@pytest.mark.parametrize("kind", ["quadratic", "exponential"])
@pytest.mark.parametrize("eta", [0.25, 0.8, 0.95])
def test_kraus_channel_matches_replacement_and_preserves_density(kind, eta):
    rng = np.random.default_rng(202609101)
    for a in [0.0, 1.0, *rng.uniform(0, 1, 10)]:
        source = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        rho = source @ source.conj().T
        rho /= np.trace(rho)
        operators = sharp.kraus_operators(a, kind, eta, beta=2.0)
        np.testing.assert_allclose(sum(k.conj().T @ k for k in operators), np.eye(2), atol=1e-14)
        population = a ** 2 if kind == "quadratic" else np.expm1(2 * a) / np.expm1(2)
        expected = eta * rho + (1 - eta) * np.diag([1 - population, population])
        actual = sharp.apply_channel(rho, a, kind, eta)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        assert abs(np.trace(actual) - 1) < 1e-14
        np.testing.assert_allclose(actual, actual.conj().T, atol=1e-14)
        assert np.linalg.eigvalsh(actual).min() >= -1e-14


@pytest.mark.parametrize("kind", ["quadratic", "exponential"])
@pytest.mark.parametrize("length", [1, 4, 32, 160])
def test_finite_prefix_channel_gaps_are_distinct_from_closure_diameter(kind, length):
    eta = 0.8
    left = right = np.diag([1.0, 0.0])
    for _ in range(length):
        left = sharp.apply_channel(left, 0.0, kind, eta)
        right = sharp.apply_channel(right, 1.0, kind, eta)
    rng = np.random.default_rng(202609102)
    for d in range(1, 9):
        value = rng.uniform()
        left = sharp.apply_channel(left, value, kind, eta)
        right = sharp.apply_channel(right, value, kind, eta)
        gap = float((right - left)[1, 1].real)
        assert gap == pytest.approx(sharp.finite_prefix_gap(d, length, eta), abs=2e-14)
        assert gap <= sharp.exact_diameter(d, eta) + 2e-14
        if length <= 32:
            assert gap < sharp.exact_diameter(d, eta)


@pytest.mark.parametrize("kind", ["quadratic", "exponential"])
@pytest.mark.parametrize("r", [1, 2, 3])
def test_small_statevectors_match_independent_binomial_sums(kind, r):
    rng = np.random.default_rng(202609103)
    for d in (1, 2, 3):
        for a in (rng.uniform(size=d), np.linspace(0, 1, d)):
            probabilities = binom.pmf(np.arange(r + 1)[:, None], r, a[None, :])
            values = sharp.target_population(np.arange(r + 1) / r, kind)
            expected = 0.8 ** d / 2 + (values @ probabilities) @ sharp.history_weights(d)
            assert sharp.quantum_readout(a, r, kind) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("a", [
    np.zeros(8), np.ones(8), np.full(8, 0.5),
    np.array([0.0, 1.0, 0.2, 0.7, 0.1, 0.9, 0.3, 0.8]),
])
def test_sixteen_qubit_factorial_readout_at_endpoints_and_mixed_inputs(a):
    prediction = sharp.quantum_readout(a, 2, readout="factorial_quadratic")
    # These interval endpoints are analytic reachable-closure values, not a
    # claim that finite all-one prefixes reach population one exactly.
    low = (0.2 * 0.8 ** np.arange(7, -1, -1)) @ (a ** 2)
    high = low + 0.8 ** 8
    assert prediction == pytest.approx((low + high) / 2, abs=2e-14)
    assert max(abs(prediction - low), abs(prediction - high)) == pytest.approx(
        0.08388608, abs=2e-14)


@pytest.mark.parametrize("r", [2, 3, 4])
def test_factorial_moment_identity_for_quadratic_readout(r):
    rng = np.random.default_rng(202609104)
    for _ in range(4):
        a = rng.uniform(size=3)
        expected = 0.8 ** 3 / 2 + sharp.history_weights(3) @ (a ** 2)
        assert sharp.quantum_readout(a, r, readout="factorial_quadratic") == pytest.approx(
            expected, abs=1e-14)


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_one_copy_lp_lower_bound_matches_separate_uniform_construction(d):
    result = sharp.one_copy_midpoint_lp(d)
    bound = (1 - 0.8 ** d) / 8
    assert result["lp_midpoint_error"] == pytest.approx(bound, abs=1e-10)
    assert result["matching_uniform_error"] == pytest.approx(bound, abs=1e-14)
    # a-a^2-1/8 ranges between -1/8 and +1/8 on [0,1]. The
    # positive weighted sum has matching extrema at vertices and center.
    for a in [*itertools.product((0.0, 1.0), repeat=d), np.full(d, 0.5)]:
        prediction = sharp.quantum_readout(a, 1, readout="optimal_r1")
        assert abs(prediction - sharp.midpoint(a)) == pytest.approx(bound, abs=1e-14)


@pytest.mark.parametrize("r", [1, 4, 16, 256])
@pytest.mark.parametrize("beta", [0.5, 2.0, 4.0])
def test_exponential_moments_and_maximum_against_direct_binomial_grid(r, beta):
    grid = np.linspace(0, 1, 1001)
    points = np.arange(r + 1)
    # Evaluate g directly, independently of the normalized implementation.
    values = np.expm1(beta * points / r) / np.expm1(beta)
    observed = values @ binom.pmf(points[:, None], r, grid[None, :])
    observed -= np.expm1(beta * grid) / np.expm1(beta)
    np.testing.assert_allclose(sharp.bernstein_bias(grid, r, beta=beta), observed, atol=8e-14)
    point, maximum = sharp.maximum_bias(r, beta=beta)
    assert 0 < point < 1
    assert maximum > 0
    assert observed.max() <= maximum + 8e-14
    assert maximum - observed.max() < 2e-6
    exact_at_maximum = values @ binom.pmf(points, r, point) - np.expm1(beta * point) / np.expm1(beta)
    assert maximum == pytest.approx(exact_at_maximum, abs=8e-14)


def test_quadratic_bernstein_bias_is_not_optimal_family_error():
    a = np.array([0.1, 0.5, 0.8])
    m = sharp.midpoint(a)
    expected_bias = sharp.history_weights(3) @ (a * (1 - a) / 2)
    assert sharp.quantum_readout(a, 2) - m == pytest.approx(expected_bias, abs=1e-14)
    assert expected_bias > 0
    assert sharp.quantum_readout(a, 2, readout="factorial_quadratic") == pytest.approx(m, abs=1e-14)


def test_convergence_api_is_deterministic_factorized_and_matches_reference_values(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("large-r table must not invoke a statevector")
    monkeypatch.setattr(sharp, "quantum_readout", forbidden)
    rows = sharp.convergence_rows()
    assert rows == sharp.convergence_rows()
    assert [row["r"] for row in rows] == list(sharp.DEFAULT_COPIES)
    expected = {1: 0.28132159, 4: 0.13823087, 16: 0.09769389,
                64: 0.08734994, 256: 0.08475276}
    for row in rows:
        assert set(row) == {"r", "qubits", "error", "floor", "excess", "maximizing_input"}
        assert row["floor"] == pytest.approx(0.08388608, abs=1e-14)
        assert row["qubits"] == 8 * row["r"]
        assert row["error"] == pytest.approx(row["floor"] + row["excess"], abs=1e-14)
        if row["r"] in expected:
            assert row["error"] == pytest.approx(expected[row["r"]], abs=5e-9)
    assert all(a["error"] > b["error"] for a, b in zip(rows, rows[1:]))
    alternate = sharp.convergence_rows(d=2, eta=0.5, beta=1, copies=(1, 4))
    assert [row["qubits"] for row in alternate] == [2, 8]
    assert all(row["floor"] == 0.125 for row in alternate)


@pytest.mark.parametrize("p", [0.0, 0.01, 0.1, 0.5, 1.0])
def test_seeded_four_branch_variance_agrees_with_exact_sampling_law(p):
    panels, branches = 100_000, 4
    result = sharp.variance_panel_diagnostic(p, seed=202609105, panels=panels)
    probability = p ** 4 + (1 - p) ** 4
    assert result["zero_variance_probability"] == probability
    se = np.sqrt(probability * (1 - probability) / panels)
    assert abs(result["zero_variance_fraction"] - probability) <= 6 * se + 1e-14
    variance = p * (1 - p)
    fourth_moment = p * (1 - p) ** 4 + (1 - p) * p ** 4
    variance_of_sample_variance = (fourth_moment - (branches - 3) / (branches - 1)
                                  * variance ** 2) / branches
    corrected_se = (0.8 ** 8) ** 2 * np.sqrt(max(0, variance_of_sample_variance) / panels)
    assert abs(result["corrected_variance"] - result["population_variance"]) <= 6 * corrected_se + 1e-14
    assert result["minimax_floor"] == pytest.approx(0.08388608, abs=1e-14)


def test_variance_seed_is_local_explicit_and_repeatable():
    first = sharp.variance_panel_diagnostic(0.1, seed=17, panels=4000)
    sharp.variance_panel_diagnostic(0.1, seed=18, panels=4000)
    assert first == sharp.variance_panel_diagnostic(0.1, seed=17, panels=4000)
    assert sharp.zero_variance_probability(0.01) == pytest.approx(0.96059602, abs=1e-14)


@pytest.mark.parametrize("seed", [None, -1, True])
def test_variance_seed_cannot_be_implicit_or_invalid(seed):
    with pytest.raises(ValueError, match="seed"):
        sharp.variance_panel_diagnostic(0.1, seed=seed)


@pytest.mark.parametrize("eta", [0.0, 1.0, -0.1, np.nan])
def test_invalid_contraction_parameters_fail(eta):
    with pytest.raises(ValueError, match="eta"):
        sharp.convergence_rows(eta=eta)


@pytest.mark.parametrize("r", [0, -1, 1.5, True])
def test_invalid_copy_counts_fail(r):
    with pytest.raises(ValueError, match="r"):
        sharp.bernstein_bias(0.5, r)


def test_invalid_targets_inputs_or_incompatible_readouts_fail():
    with pytest.raises(ValueError, match="kind"):
        sharp.target_population(0.5, "typo")
    with pytest.raises(ValueError, match="inputs"):
        sharp.midpoint([1.01])
    with pytest.raises(ValueError, match="beta"):
        sharp.maximum_bias(2, beta=0)
    with pytest.raises(ValueError, match="nonempty"):
        sharp.midpoint([])
    with pytest.raises(ValueError, match="max_qubits"):
        sharp.quantum_readout([0.5] * 8, 256)
    with pytest.raises(ValueError, match="factorial_quadratic"):
        sharp.quantum_readout([0.5], 1, readout="factorial_quadratic")
    with pytest.raises(ValueError, match="optimal_r1"):
        sharp.quantum_readout([0.5], 2, readout="optimal_r1")
