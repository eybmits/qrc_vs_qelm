"""Exact-expectation checks for a separate, analytically solvable window model.

This post-hoc verification module does not read or modify experimental cohorts.
The target is a one-qubit replacement channel, equivalent on diagonal states to
a classical scalar leaky integrator. Readouts are specified analytically, not
learned. Global errors below use the derivation, not sampled numerical maxima.
"""

from __future__ import annotations

import itertools
import operator

import numpy as np
from scipy.optimize import brentq, linprog

DEFAULT_COPIES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _integer(value, name, minimum=1):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if isinstance(value, (bool, np.bool_)) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _eta(value):
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("eta must be finite and strictly between zero and one")
    return value


def _beta(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("beta must be positive and finite")
    return value


def _inputs(value):
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)) or np.any((value < 0) | (value > 1)):
        raise ValueError("inputs must be finite and in [0, 1]")
    return value


def _window(value):
    value = _inputs(value)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("window must be a nonempty one-dimensional sequence")
    return value


def target_population(a, kind="quadratic", beta=2.0):
    """Population g(a); exponential normalization is stable at both endpoints."""
    a = _inputs(a)
    if kind == "quadratic":
        return a * a
    if kind != "exponential":
        raise ValueError("kind must be 'quadratic' or 'exponential'")
    beta = _beta(beta)
    return np.exp(beta * (a - 1)) * (-np.expm1(-beta * a)) / (-np.expm1(-beta))


def history_weights(d, eta=0.8):
    """Oldest-to-newest suffix weights in the scalar population recurrence."""
    d, eta = _integer(d, "d"), _eta(eta)
    return (1 - eta) * eta ** np.arange(d - 1, -1, -1)


def exact_diameter(d, eta=0.8):
    """Supremum over arbitrary-length prehistories, including closure limits."""
    return _eta(eta) ** _integer(d, "d")


def finite_prefix_gap(d, prefix_length, eta=0.8):
    """Gap from all-zero/all-one finite prefixes followed by a shared suffix."""
    eta = _eta(eta)
    return exact_diameter(d, eta) * (1 - eta ** _integer(prefix_length, "prefix_length"))


def midpoint(window, kind="quadratic", eta=0.8, beta=2.0):
    """Midpoint of the compatible output interval for the specified window."""
    a = _window(window)
    return float(exact_diameter(len(a), eta) / 2
                 + history_weights(len(a), eta) @ target_population(a, kind, beta))


def kraus_operators(a, kind="quadratic", eta=0.8, beta=2.0):
    """Kraus realization of eta*rho + (1-eta)*diag(1-g(a), g(a))."""
    eta = _eta(eta)
    a = _inputs(a)
    if a.ndim != 0:
        raise ValueError("channel input must be scalar")
    population = float(target_population(a, kind, beta))
    probabilities = (1 - population, population)
    operators = [np.sqrt(eta) * np.eye(2)]
    for row, col in itertools.product(range(2), repeat=2):
        item = np.zeros((2, 2))
        item[row, col] = np.sqrt((1 - eta) * probabilities[row])
        operators.append(item)
    return tuple(operators)


def apply_channel(rho, a, kind="quadratic", eta=0.8, beta=2.0):
    """Apply explicit Kraus operators; inputs must have a finite 2x2 matrix."""
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2) or not np.all(np.isfinite(rho)):
        raise ValueError("rho must be a finite 2x2 matrix")
    return sum(k @ rho @ k.conj().T for k in kraus_operators(a, kind, eta, beta))


def quantum_readout(window, r, kind="quadratic", eta=0.8, beta=2.0,
                    readout="bernstein", max_qubits=16):
    """Enumerate every amplitude and basis outcome of a small product register.

    This deliberately avoids binomial-moment formulas. The exponential-size
    operation is bounded explicitly; high-r convergence uses factorization.
    ``factorial_quadratic`` uses unbiased second factorial moments (r >= 2).
    ``optimal_r1`` is restricted to one-copy computational count features.
    """
    a, r = _window(window), _integer(r, "r")
    eta = _eta(eta)
    max_qubits = _integer(max_qubits, "max_qubits")
    n_qubits = len(a) * r
    if n_qubits > max_qubits:
        raise ValueError("explicit statevector exceeds max_qubits")
    if readout not in {"bernstein", "factorial_quadratic", "optimal_r1"}:
        raise ValueError("unknown readout")
    target_population(a, kind, beta)  # Validate the declared target as well.
    if readout == "factorial_quadratic" and (kind != "quadratic" or r < 2):
        raise ValueError("factorial_quadratic requires quadratic target and r >= 2")
    if readout == "optimal_r1" and (kind != "quadratic" or r != 1):
        raise ValueError("optimal_r1 requires quadratic target and r = 1")
    psi = np.array([1.0])
    for value in a:
        for _ in range(r):
            psi = np.kron(psi, [np.sqrt(1 - value), np.sqrt(value)])
    ids = np.arange(len(psi), dtype=np.uint64)
    bit_positions = np.arange(n_qubits - 1, -1, -1, dtype=np.uint64)
    counts = ((ids[:, None] >> bit_positions) & 1).reshape(-1, len(a), r).sum(axis=2)
    counts = counts.astype(float)
    if readout == "factorial_quadratic":
        values = counts * (counts - 1) / (r * (r - 1))
    elif readout == "optimal_r1":
        values = counts - 1 / 8
    else:
        values = target_population(counts / r, kind, beta)
    coefficients = exact_diameter(len(a), eta) / 2 + values @ history_weights(len(a), eta)
    return float((np.abs(psi) ** 2) @ coefficients)


def bernstein_bias(a, r, kind="exponential", beta=2.0):
    """E[g(K/r)] - g(a), K~Binomial(r,a), without binomial enumeration."""
    a, r = _inputs(a), _integer(r, "r")
    if kind == "quadratic":
        return a * (1 - a) / r
    if kind != "exponential":
        raise ValueError("kind must be 'quadratic' or 'exponential'")
    beta = _beta(beta)
    if beta / r < 50:
        log_moment = r * np.log1p(a * np.expm1(beta / r))
    else:
        with np.errstate(divide="ignore"):
            log_moment = r * np.logaddexp(np.log1p(-a), np.log(a) + beta / r)
    expectation = (np.exp(log_moment - beta) * (-np.expm1(-log_moment))
                   / (-np.expm1(-beta)))
    return expectation - target_population(a, kind, beta)


def maximum_bias(r, kind="exponential", beta=2.0):
    """Return the analytically characterized maximizing input and bias.

    For positive beta the log ratio of derivative terms is concave (linear
    for r=1), positive at zero and negative at one, with exactly one zero there.
    Root finding locates this critical point; a grid is not the certificate.
    """
    r = _integer(r, "r")
    if kind == "quadratic":
        return 0.5, 1 / (4 * r)
    if kind != "exponential":
        raise ValueError("kind must be 'quadratic' or 'exponential'")
    beta = _beta(beta)
    step = beta / r
    log_c = step + np.log(-np.expm1(-step))

    def derivative_log_ratio(a):
        if step < 50:
            log_mix = np.log1p(a * np.expm1(step))
        else:
            with np.errstate(divide="ignore"):
                log_mix = np.logaddexp(np.log1p(-a), np.log(a) + step)
        return np.log(r / beta) + log_c + (r - 1) * log_mix - beta * a

    point = float(brentq(derivative_log_ratio, 0, 1, xtol=1e-14))
    return point, float(bernstein_bias(point, r, kind, beta))


def convergence_rows(d=8, eta=0.8, beta=2.0, copies=DEFAULT_COPIES):
    """Plot-ready exponential-target construction errors, not fitted-model errors.

    Every row has r, qubits, error, floor, excess, and maximizing_input keys.
    The maximum factorizes because the bias and suffix weights are nonnegative.
    No large-register statevector is constructed by this function.
    """
    d, eta, beta = _integer(d, "d"), _eta(eta), _beta(beta)
    floor = exact_diameter(d, eta) / 2
    rows = []
    for r in copies:
        r = _integer(r, "r")
        point, bias = maximum_bias(r, beta=beta)
        excess = (1 - eta ** d) * bias
        rows.append({"r": r, "qubits": d * r, "error": floor + excess,
                     "floor": floor, "excess": excess, "maximizing_input": point})
    return rows


def one_copy_midpoint_lp(d, eta=0.8):
    """Finite LP lower bound for the quadratic computational count-feature family.

    Epsilon bounds the midpoint approximation on vertices and the cube center.
    A separate uniform upper bound, from replacing a^2 by a-1/8, matches it.
    This is not an optimization over arbitrary quantum encodings or observables.
    """
    d, eta = _integer(d, "d"), _eta(eta)
    if d > 8:
        raise ValueError("explicit LP is limited to d <= 8")
    vertices = np.array(list(itertools.product((0.0, 1.0), repeat=d)))
    points = np.vstack((vertices, np.full((1, d), 0.5)))
    features = np.prod(np.where(vertices[None, :, :] == 1,
                                points[:, None, :], 1 - points[:, None, :]), axis=2)
    outputs = np.array([midpoint(a, eta=eta) for a in points])
    n_coefficients = len(vertices)
    result = linprog(np.r_[np.zeros(n_coefficients), 1.0],
                     A_ub=np.vstack((np.c_[features, -np.ones(len(points))],
                                     np.c_[-features, -np.ones(len(points))])),
                     b_ub=np.r_[outputs, -outputs],
                     bounds=[(None, None)] * n_coefficients + [(0, None)], method="highs")
    if not result.success:
        raise RuntimeError(f"one-copy LP failed: {result.message}")
    return {"lp_midpoint_error": float(result.fun),
            "matching_uniform_error": (1 - eta ** d) / 8,
            "constraints": 2 * len(points)}


def zero_variance_probability(p, branches=4):
    """Probability all independently sampled extreme old states are identical."""
    p = _inputs(p)
    if p.ndim != 0:
        raise ValueError("p must be scalar")
    branches = _integer(branches, "branches", 2)
    return float(p ** branches + (1 - p) ** branches)


def variance_panel_diagnostic(p, *, seed, d=8, eta=0.8, panels=100_000, branches=4):
    """Seeded distribution check, holding the admissible history set unchanged.

    Old populations are closure endpoints 0 and 1. Sampling probabilities do
    not redefine the history set over which exact_diameter takes its supremum.
    The correction is unbiased for variance, not for its square root.
    """
    probability = zero_variance_probability(p, branches)
    panels, branches = _integer(panels, "panels"), _integer(branches, "branches", 2)
    seed = _integer(seed, "seed", 0)
    diameter = exact_diameter(d, eta)
    old = np.random.default_rng(seed).binomial(1, float(p), size=(panels, branches))
    sample_variance = np.var(diameter * old, axis=1, ddof=0)
    return {"panels": panels, "branches": branches,
            "zero_variance_fraction": float(np.mean(sample_variance == 0)),
            "zero_variance_probability": probability,
            "corrected_variance": float(np.mean(sample_variance) * branches / (branches - 1)),
            "population_variance": diameter ** 2 * float(p) * (1 - float(p)),
            "minimax_floor": diameter / 2}
