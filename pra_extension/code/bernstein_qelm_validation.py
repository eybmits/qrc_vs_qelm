"""Constructive Bernstein-QELM validation for contractive QRC replay.

The construction is deliberately explicit.  A suffix
``x = (x_1, ..., x_d)`` is encoded using ``r`` copies of

    |psi(x_j)> = sqrt(1-x_j)|0> + sqrt(x_j)|1>

for every lag ``j``.  Measuring the Hamming weight ``k_j`` of each block has
the product-binomial probability

    p_k(x) = product_j C(r,k_j) x_j**k_j (1-x_j)**(r-k_j).

A linear readout with weight ``g(k/r)`` therefore returns the tensor Bernstein
polynomial ``B_r g`` exactly.  For a Euclidean ``L``-Lipschitz target,

    |B_r g(x) - g(x)| <= L sqrt(d / (4 r)).

The one-qubit QRC below is not a new simulator.  It is the smallest
specialisation of the repository's existing input-affine ``LindbladReservoir``
with local loss.  It makes the contraction and Lipschitz certificates fully
auditable while keeping the validation inexpensive.
"""

from __future__ import annotations

from itertools import product
from math import comb, exp, sqrt
from typing import Callable, Iterable, Sequence

import numpy as np

from qrc_vs_qelm.core import dissipators, liouvillian, operators, reservoirs


ArrayFunction = Callable[[np.ndarray], np.ndarray]


def _as_points(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("points must have shape (n_points, d) with d >= 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite")
    if np.min(values) < -1e-14 or np.max(values) > 1.0 + 1e-14:
        raise ValueError("points must lie in [0, 1]^d")
    return np.clip(values, 0.0, 1.0)


def multi_indices(d: int, r: int) -> np.ndarray:
    """Return ``{0, ..., r}^d`` in deterministic lexicographic order."""

    if d < 1 or r < 1:
        raise ValueError("d and r must be positive")
    return np.asarray(list(product(range(r + 1), repeat=d)), dtype=int)


def bernstein_nodes(d: int, r: int) -> np.ndarray:
    """Return the readout interpolation nodes ``k/r``."""

    return multi_indices(d, r).astype(float) / float(r)


def one_dimensional_binomial_basis(x: np.ndarray, r: int) -> np.ndarray:
    """Return ``b_{r,k}(x)`` for all samples and ``k=0,...,r``."""

    if r < 1:
        raise ValueError("r must be positive")
    values = np.asarray(x, dtype=float).reshape(-1)
    if np.min(values) < -1e-14 or np.max(values) > 1.0 + 1e-14:
        raise ValueError("x must lie in [0, 1]")
    values = np.clip(values, 0.0, 1.0)
    ks = np.arange(r + 1, dtype=int)
    coefficients = np.asarray([comb(r, int(k)) for k in ks], dtype=float)
    # NumPy defines 0**0 as 1, which is exactly the endpoint convention needed
    # by the binomial basis.
    return (
        coefficients[None, :]
        * values[:, None] ** ks[None, :]
        * (1.0 - values[:, None]) ** (r - ks)[None, :]
    )


def tensor_bernstein_probabilities(
    points: np.ndarray | Sequence[Sequence[float]], r: int
) -> np.ndarray:
    """Return all joint Hamming-weight probabilities for each suffix."""

    values = _as_points(points)
    indices = multi_indices(values.shape[1], r)
    probabilities = np.ones((values.shape[0], indices.shape[0]), dtype=float)
    for coordinate in range(values.shape[1]):
        basis = one_dimensional_binomial_basis(values[:, coordinate], r)
        probabilities *= basis[:, indices[:, coordinate]]
    return probabilities


def quantum_hamming_probabilities(x: Sequence[float], r: int) -> np.ndarray:
    """Compute block-Hamming probabilities directly from the statevector.

    This routine is intentionally exponential and is used only for the small
    exact quantum audit.  Its output order matches :func:`multi_indices`.
    """

    values = _as_points(np.asarray(x, dtype=float))[0]
    n_qubits = int(values.size * r)
    if n_qubits > 20:
        raise ValueError("direct statevector audit is limited to 20 qubits")
    state = np.asarray([1.0 + 0.0j])
    for value in values:
        qubit = np.asarray([sqrt(1.0 - value), sqrt(value)], dtype=complex)
        for _ in range(r):
            state = np.kron(state, qubit)

    grouped = {tuple(index): 0.0 for index in multi_indices(values.size, r)}
    for basis_index, probability in enumerate(np.abs(state) ** 2):
        bits = format(basis_index, f"0{n_qubits}b")
        weights = tuple(
            bits[block * r : (block + 1) * r].count("1")
            for block in range(values.size)
        )
        grouped[weights] += float(probability)
    return np.asarray(
        [grouped[tuple(index)] for index in multi_indices(values.size, r)],
        dtype=float,
    )


def bernstein_from_node_values(
    points: np.ndarray | Sequence[Sequence[float]],
    r: int,
    node_values: Sequence[float],
) -> np.ndarray:
    """Evaluate the QELM readout ``sum_k g(k/r) p_k(x)``."""

    probabilities = tensor_bernstein_probabilities(points, r)
    weights = np.asarray(node_values, dtype=float).reshape(-1)
    if weights.size != probabilities.shape[1]:
        raise ValueError("one readout weight is required for every Hamming outcome")
    return probabilities @ weights


def bernstein_approximation(
    points: np.ndarray | Sequence[Sequence[float]], r: int, target: ArrayFunction
) -> np.ndarray:
    """Evaluate ``B_r target`` using the constructive QELM weights."""

    values = _as_points(points)
    nodes = bernstein_nodes(values.shape[1], r)
    node_values = np.asarray(target(nodes), dtype=float).reshape(-1)
    return bernstein_from_node_values(values, r, node_values)


def bernstein_lipschitz_bound(lipschitz_constant: float, d: int, r: int) -> float:
    """Return ``L sqrt(d/(4r))`` for the tensor Bernstein construction."""

    if lipschitz_constant < 0.0 or d < 1 or r < 1:
        raise ValueError("L must be nonnegative and d,r must be positive")
    return float(lipschitz_constant * sqrt(d / (4.0 * r)))


def bernstein_coordinate_bound(
    coordinate_constants: Sequence[float], r: int
) -> float:
    """Return the sharper ``sum_i c_i/(2 sqrt(r))`` certificate."""

    constants = np.asarray(coordinate_constants, dtype=float).reshape(-1)
    if constants.size < 1 or np.min(constants) < 0.0 or r < 1:
        raise ValueError("coordinate constants must be nonnegative and r positive")
    return float(np.sum(constants) / (2.0 * sqrt(r)))


def cartesian_grid(d: int, points_per_axis: int) -> np.ndarray:
    """Deterministic closed grid on ``[0,1]^d``."""

    if d < 1 or points_per_axis < 2:
        raise ValueError("d must be positive and points_per_axis at least two")
    axis = np.linspace(0.0, 1.0, points_per_axis)
    return np.asarray(list(product(axis, repeat=d)), dtype=float)


def distance_target(points: np.ndarray) -> np.ndarray:
    """A nonlinear, nonsmooth, exactly 1-Lipschitz reference target."""

    values = _as_points(points)
    center = np.linspace(0.25, 0.75, values.shape[1])
    return np.linalg.norm(values - center[None, :], axis=1)


def build_contracting_qrc(
    *, gamma: float = 4.0, dt: float = 0.5, h: float = 0.5, h_in: float = 1.0
) -> reservoirs.LindbladReservoir:
    """Build a one-qubit instance of the existing local-loss QRC family."""

    if gamma <= 0.0 or dt <= 0.0:
        raise ValueError("gamma and dt must be positive")
    return reservoirs.LindbladReservoir.from_terms(
        n_qubits=1,
        H_static=float(h) * operators.SZ,
        H_drive=float(h_in) * operators.SX,
        jumps=dissipators.local_loss(1, float(gamma)),
        dt=float(dt),
        cache_propagators=True,
    )


def qrc_suffix_output(
    points: np.ndarray | Sequence[Sequence[float]],
    model: reservoirs.LindbladReservoir,
    observable: np.ndarray = operators.SZ,
) -> np.ndarray:
    """Replay each suffix from the QRC reset state and measure ``observable``."""

    values = _as_points(points)
    outputs = np.empty(values.shape[0], dtype=float)
    for row, suffix in enumerate(values):
        state = model.initial_state()
        for value in suffix:
            state = model.step(state, float(value))
        outputs[row] = float(np.trace(observable @ state).real)
    return outputs


def _bloch_matrix_from_map(action: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    paulis = (operators.SX, operators.SY, operators.SZ)
    matrix = np.empty((3, 3), dtype=float)
    for column, pauli in enumerate(paulis):
        image = action(pauli)
        for row, test_pauli in enumerate(paulis):
            matrix[row, column] = float(
                0.5 * np.trace(test_pauli @ image).real
            )
    return matrix


def bloch_generator_matrix(
    model: reservoirs.LindbladReservoir, input_value: float
) -> np.ndarray:
    """Return the linear Bloch generator on traceless Hermitian differences."""

    generator = model.liouvillian(float(input_value))

    def action(matrix: np.ndarray) -> np.ndarray:
        return liouvillian.unvec(generator @ liouvillian.vec(matrix), 2)

    return _bloch_matrix_from_map(action)


def bloch_channel_matrix(
    model: reservoirs.LindbladReservoir, input_value: float
) -> np.ndarray:
    """Return the one-step linear Bloch map on state differences."""

    # Reuse the reservoir's own cached propagator so that this audit exercises
    # exactly the discrete channel used by ``model.step``.
    propagator = model._sub_propagator(float(input_value), model.dt)

    def action(matrix: np.ndarray) -> np.ndarray:
        return liouvillian.apply_propagator(propagator, matrix)

    return _bloch_matrix_from_map(action)


def contraction_audit(
    model: reservoirs.LindbladReservoir,
    *,
    gamma: float,
    input_grid: Iterable[float] = np.linspace(0.0, 1.0, 101),
) -> dict[str, float]:
    """Audit the analytic uniform trace-distance contraction certificate.

    For one-qubit local loss, the symmetric part of the Bloch generator is
    ``diag(-gamma/2, -gamma/2, -gamma)``.  Any Hamiltonian contributes only a
    skew-symmetric part.  Hence the Euclidean logarithmic norm is at most
    ``-gamma/2`` and the trace-distance contraction factor is uniformly
    ``eta = exp(-gamma*dt/2)``.
    """

    values = np.asarray(list(input_grid), dtype=float)
    log_norms = []
    channel_norms = []
    for value in values:
        generator = bloch_generator_matrix(model, float(value))
        symmetric = 0.5 * (generator + generator.T)
        log_norms.append(float(np.linalg.eigvalsh(symmetric)[-1]))
        channel_norms.append(
            float(np.linalg.svd(bloch_channel_matrix(model, float(value)), compute_uv=False)[0])
        )
    eta = exp(-float(gamma) * model.dt / 2.0)
    return {
        "eta_certificate": float(eta),
        "log_norm_certificate": float(-float(gamma) / 2.0),
        "max_sampled_generator_log_norm": float(max(log_norms)),
        "max_sampled_channel_trace_contraction": float(max(channel_norms)),
        "sampled_input_count": int(values.size),
    }


def qrc_suffix_lipschitz_certificate(
    *,
    d: int,
    gamma: float,
    dt: float,
    h_in: float,
    observable_norm: float = 1.0,
) -> dict[str, object]:
    """Certify the Euclidean Lipschitz constant of a length-``d`` replay map.

    Duhamel's formula gives an input-channel difference at most
    ``2 dt ||H_drive|| |delta x|`` in trace norm.  The subsequent common
    channels contribute one factor ``eta`` each.  Cauchy--Schwarz then gives
    the Euclidean suffix Lipschitz constant.
    """

    if d < 1:
        raise ValueError("d must be positive")
    eta = exp(-float(gamma) * float(dt) / 2.0)
    base = 2.0 * float(dt) * abs(float(h_in)) * float(observable_norm)
    coordinate_constants = [base * eta ** (d - 1 - lag) for lag in range(d)]
    return {
        "eta": float(eta),
        "coordinate_constants_oldest_to_newest": [
            float(value) for value in coordinate_constants
        ],
        "euclidean_lipschitz_constant": float(
            np.linalg.norm(coordinate_constants)
        ),
    }


def error_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    difference = np.asarray(estimate, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def validate_feature_probabilities() -> dict[str, object]:
    """Run partition-of-unity and direct statevector probability audits."""

    audit_points = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.13, 0.71], [0.5, 0.5]]
    )
    partition = tensor_bernstein_probabilities(audit_points, r=5)
    analytic = tensor_bernstein_probabilities([[0.17, 0.68]], r=3)[0]
    quantum = quantum_hamming_probabilities([0.17, 0.68], r=3)
    return {
        "partition_r": 5,
        "partition_points": int(audit_points.shape[0]),
        "max_partition_error": float(np.max(np.abs(np.sum(partition, axis=1) - 1.0))),
        "min_probability": float(np.min(partition)),
        "quantum_audit_d": 2,
        "quantum_audit_r": 3,
        "quantum_audit_qubits": 6,
        "max_quantum_vs_binomial_error": float(np.max(np.abs(quantum - analytic))),
        "quantum_probability_sum_error": float(abs(np.sum(quantum) - 1.0)),
    }


def generic_bernstein_convergence(
    r_values: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> list[dict[str, object]]:
    """Validate the theorem on a nonlinear exactly 1-Lipschitz target."""

    d = 2
    points = cartesian_grid(d, 51)
    reference = distance_target(points)
    rows: list[dict[str, object]] = []
    for r in r_values:
        estimate = bernstein_approximation(points, int(r), distance_target)
        metrics = error_metrics(reference, estimate)
        bound = bernstein_lipschitz_bound(1.0, d, int(r))
        rows.append(
            {
                "target": "distance_to_fixed_point",
                "d": d,
                "r": int(r),
                "qelm_qubits": int(d * r),
                "feature_outcomes": int((r + 1) ** d),
                "lipschitz_constant": 1.0,
                "certified_max_error_bound": float(bound),
                **metrics,
                "max_error_to_bound_ratio": float(metrics["max_abs_error"] / bound),
            }
        )
    return rows


def qrc_bernstein_convergence(
    model: reservoirs.LindbladReservoir,
    *,
    d: int = 2,
    gamma: float = 4.0,
    dt: float = 0.5,
    h_in: float = 1.0,
    r_values: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> list[dict[str, object]]:
    """Observe constructive-QELM convergence on the QRC suffix replay map."""

    points = cartesian_grid(d, 41)
    reference = qrc_suffix_output(points, model)
    certificate = qrc_suffix_lipschitz_certificate(
        d=d, gamma=gamma, dt=dt, h_in=h_in
    )
    lipschitz = float(certificate["euclidean_lipschitz_constant"])
    rows: list[dict[str, object]] = []
    for r in r_values:
        nodes = bernstein_nodes(d, int(r))
        node_values = qrc_suffix_output(nodes, model)
        estimate = bernstein_from_node_values(points, int(r), node_values)
        metrics = error_metrics(reference, estimate)
        bound = bernstein_coordinate_bound(
            certificate["coordinate_constants_oldest_to_newest"], int(r)
        )
        rows.append(
            {
                "target": "contractive_qrc_suffix_replay_z",
                "d": int(d),
                "r": int(r),
                "qelm_qubits": int(d * r),
                "feature_outcomes": int((r + 1) ** d),
                "lipschitz_constant": float(lipschitz),
                "certified_max_error_bound": float(bound),
                **metrics,
                "max_error_to_bound_ratio": float(metrics["max_abs_error"] / bound),
            }
        )
    return rows


def qrc_history_convergence(
    model: reservoirs.LindbladReservoir,
    *,
    gamma: float = 4.0,
    max_window: int = 8,
    sequence_length: int = 257,
    seed: int = 26_082_944,
) -> list[dict[str, object]]:
    """Compare persistent QRC outputs with reset-and-replay suffix outputs."""

    rng = np.random.default_rng(seed)
    inputs = rng.uniform(0.0, 1.0, size=sequence_length)
    state = model.initial_state()
    persistent = np.empty(sequence_length, dtype=float)
    for index, value in enumerate(inputs):
        state = model.step(state, float(value))
        persistent[index] = float(np.trace(operators.SZ @ state).real)

    eta = exp(-float(gamma) * model.dt / 2.0)
    rows: list[dict[str, object]] = []
    for d in range(1, max_window + 1):
        suffixes = np.asarray(
            [inputs[index - d + 1 : index + 1] for index in range(d, sequence_length)]
        )
        replay = qrc_suffix_output(suffixes, model)
        metrics = error_metrics(persistent[d:], replay)
        bound = 2.0 * eta**d
        rows.append(
            {
                "d": int(d),
                "evaluated_times": int(replay.size),
                "eta": float(eta),
                "certified_max_history_error_bound": float(bound),
                **metrics,
                "max_error_to_bound_ratio": float(metrics["max_abs_error"] / bound),
            }
        )
    return rows


def run_validation() -> dict[str, object]:
    """Run the complete deterministic validation and return serialisable data."""

    gamma = 4.0
    dt = 0.5
    h = 0.5
    h_in = 1.0
    d = 2
    r_values = (1, 2, 4, 8, 16, 32)
    model = build_contracting_qrc(gamma=gamma, dt=dt, h=h, h_in=h_in)
    feature_audit = validate_feature_probabilities()
    contraction = contraction_audit(model, gamma=gamma)
    lipschitz = qrc_suffix_lipschitz_certificate(
        d=d, gamma=gamma, dt=dt, h_in=h_in
    )
    generic_rows = generic_bernstein_convergence(r_values)
    qrc_rows = qrc_bernstein_convergence(
        model,
        d=d,
        gamma=gamma,
        dt=dt,
        h_in=h_in,
        r_values=r_values,
    )
    history_rows = qrc_history_convergence(model, gamma=gamma)
    return {
        "schema_version": "bernstein_qelm_validation_v1",
        "construction": {
            "input_domain": "[0,1]^d",
            "encoding": "r copies of sqrt(1-x_j)|0>+sqrt(x_j)|1> per lag j",
            "measurement": "joint block Hamming weights",
            "readout_weights": "g_d(k/r)",
            "implemented_function": "tensor Bernstein polynomial B_r g_d",
            "uniform_error_bound": (
                "sum_i c_i/(2 sqrt(r)); L_g sqrt(d/(4r)) is a generic fallback"
            ),
            "resource_note": "d*r qubits and (r+1)^d joint outcomes",
        },
        "qrc_configuration": {
            "implementation": "qrc_vs_qelm.core.reservoirs.LindbladReservoir",
            "n_qubits": 1,
            "dissipator": "local_loss",
            "gamma": gamma,
            "dt": dt,
            "h_z": h,
            "h_in_x": h_in,
            "observable": "Pauli Z",
            "replay_window": d,
        },
        "feature_probability_audit": feature_audit,
        "contraction_audit": contraction,
        "qrc_lipschitz_certificate": lipschitz,
        "generic_convergence": generic_rows,
        "qrc_bernstein_convergence": qrc_rows,
        "qrc_history_convergence": history_rows,
    }
