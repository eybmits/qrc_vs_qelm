"""Observable-memory diagnostics outside the frozen canonical material set.

The module reuses the frozen QRC simulator read-only.  It measures the exact
conditional risk of measured teacher outputs under common suffixes and compares
that risk with two transfer-operator calculations:

1. an exact covariance propagation from the prefix-state ensemble; and
2. rigorous Hilbert--Schmidt-to-observable upper bounds, first on the full
   trace-zero operator space and then on the sampled prefix-difference span.

The second calculation is "observable restricted" because the transfer product
is composed with the selected Pauli readout before its induced norm is taken.
It is a suffix-specific finite-dimensional bound, not a claim of uniform
trace-norm contraction or an asymptotic theorem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from qrc_vs_qelm.core import dissipators, liouvillian, readout, reservoirs


_DISSIPATOR_BUILDERS = {
    "local_loss": dissipators.local_loss,
    "dephasing": dissipators.dephasing,
    "collective_loss": dissipators.collective_loss,
}


@dataclass(frozen=True)
class RootSpec:
    """Fresh deterministic seed block for one paired stress-test replicate."""

    root_id: str
    coupling_seeds: Mapping[int, int]
    prefix_seed: int
    suffix_seed: int


@dataclass(frozen=True)
class ObservablePanel:
    """Named measured-output projection."""

    name: str
    axes: tuple[str, ...]
    max_weight: int


@dataclass(frozen=True)
class StressConfig:
    """Complete deterministic configuration for the extension harness."""

    roots: tuple[RootSpec, ...]
    n_qubits: tuple[int, ...]
    dissipator_families: tuple[str, ...]
    gammas: tuple[float, ...]
    windows: tuple[int, ...]
    observable_panels: tuple[ObservablePanel, ...]
    quantize_levels: int | None
    suffix_groups: int
    prefix_branches: int
    prefix_burnin: int
    dt: float = 0.5
    h: float = 0.5
    h_in: float = 1.0
    coupling_scale: float = 1.0


def default_config(profile: str = "full") -> StressConfig:
    """Return a deterministic quantized or continuous stress configuration."""

    panels = (
        ObservablePanel("canonical_xyz_w2", ("x", "y", "z"), 2),
        ObservablePanel("coherence_xy_w1", ("x", "y"), 1),
        ObservablePanel("population_z_w1", ("z",), 1),
    )
    if profile == "smoke":
        return StressConfig(
            roots=(
                RootSpec(
                    "pra_smoke01",
                    {2: 9_101_012},
                    prefix_seed=9_111_011,
                    suffix_seed=9_121_011,
                ),
            ),
            n_qubits=(2,),
            dissipator_families=("local_loss", "dephasing", "collective_loss"),
            gammas=(1.0,),
            windows=(1, 2),
            observable_panels=panels,
            quantize_levels=5,
            suffix_groups=4,
            prefix_branches=3,
            prefix_burnin=8,
        )
    if profile == "continuous":
        continuous_roots = tuple(
            RootSpec(
                root_id=f"cont{index:02d}",
                coupling_seeds={
                    2: 9_300_000 + 100 * index + 2,
                    3: 9_300_000 + 100 * index + 3,
                },
                prefix_seed=9_310_000 + 100 * index + 1,
                suffix_seed=9_320_000 + 100 * index + 1,
            )
            for index in range(1, 13)
        )
        return StressConfig(
            roots=continuous_roots,
            n_qubits=(2, 3),
            dissipator_families=("local_loss", "dephasing", "collective_loss"),
            gammas=(4.0,),
            windows=(1, 4, 8),
            observable_panels=panels,
            quantize_levels=None,
            suffix_groups=6,
            prefix_branches=4,
            prefix_burnin=16,
        )
    if profile != "full":
        raise ValueError("profile must be 'smoke', 'full', or 'continuous'")
    roots = tuple(
        RootSpec(
            root_id=f"pra{index:02d}",
            coupling_seeds={
                2: 9_100_000 + 100 * index + 2,
                3: 9_100_000 + 100 * index + 3,
                4: 9_100_000 + 100 * index + 4,
            },
            prefix_seed=9_110_000 + 100 * index + 1,
            suffix_seed=9_120_000 + 100 * index + 1,
        )
        # The four pra01--pra04 roots were used only for the first exploratory
        # calibration pass.  The paper-grade stress test uses twelve entirely
        # fresh roots after the analysis code and estimands were fixed.
        for index in range(5, 17)
    )
    return StressConfig(
        roots=roots,
        n_qubits=(2, 3, 4),
        dissipator_families=("local_loss", "dephasing", "collective_loss"),
        gammas=(0.25, 1.0, 4.0),
        windows=(1, 2, 4, 8),
        observable_panels=panels,
        quantize_levels=13,
        suffix_groups=12,
        prefix_branches=4,
        prefix_burnin=48,
    )


def validate_config(config: StressConfig) -> None:
    """Reject ambiguous panels, incomplete seeds, and unsafe factorial axes."""

    if not config.roots:
        raise ValueError("at least one root is required")
    root_ids = [root.root_id for root in config.roots]
    if len(set(root_ids)) != len(root_ids) or any(not value for value in root_ids):
        raise ValueError("root ids must be nonempty and unique")
    if not config.n_qubits or any(int(n) < 2 for n in config.n_qubits):
        raise ValueError("n_qubits must contain integers >= 2")
    if len(set(config.n_qubits)) != len(config.n_qubits):
        raise ValueError("n_qubits must be unique")
    if set(config.dissipator_families) - set(_DISSIPATOR_BUILDERS):
        raise ValueError("unknown dissipator family")
    if not config.gammas or any(not np.isfinite(g) or g <= 0 for g in config.gammas):
        raise ValueError("gammas must be finite and positive")
    if not config.windows or any(int(d) < 1 for d in config.windows):
        raise ValueError("windows must be positive")
    if tuple(sorted(set(config.windows))) != config.windows:
        raise ValueError("windows must be strictly increasing")
    if config.quantize_levels is not None:
        if config.quantize_levels < 2:
            raise ValueError("quantize_levels must be at least two or None")
        if not 1 <= config.suffix_groups <= config.quantize_levels:
            raise ValueError("suffix_groups must lie in [1, quantize_levels]")
    elif config.suffix_groups < 1:
        raise ValueError("suffix_groups must be positive for continuous inputs")
    if config.prefix_branches < 2 or config.prefix_burnin < 1:
        raise ValueError("prefix_branches >= 2 and prefix_burnin >= 1 are required")
    if any(not np.isfinite(x) for x in (config.dt, config.h, config.h_in)):
        raise ValueError("Hamiltonian parameters must be finite")
    if config.dt <= 0 or config.coupling_scale <= 0:
        raise ValueError("dt and coupling_scale must be positive")
    panel_names = [panel.name for panel in config.observable_panels]
    if not panel_names or len(set(panel_names)) != len(panel_names):
        raise ValueError("observable panel names must be nonempty and unique")
    for root in config.roots:
        if set(root.coupling_seeds) != set(config.n_qubits):
            raise ValueError(f"coupling seed coverage mismatch for {root.root_id}")
        values = list(root.coupling_seeds.values()) + [root.prefix_seed, root.suffix_seed]
        if len(set(values)) != len(values) or any(int(value) < 0 for value in values):
            raise ValueError(f"seeds must be nonnegative and distinct in {root.root_id}")


def normalized_jump_family(
    family: str,
    n_qubits: int,
    gamma: float,
) -> tuple[list[tuple[np.ndarray, float]], dict[str, float]]:
    """Return a jump family matched to local-loss Hilbert--Schmidt strength.

    ``gamma`` names the reference local-loss rate.  Pure dephasing is therefore
    rescaled (its bare Pauli-Z jumps have twice the squared Hilbert--Schmidt norm
    of lowering jumps), while collective loss already has the matching norm by
    construction.
    """

    if family not in _DISSIPATOR_BUILDERS:
        raise ValueError(f"unknown dissipator family: {family!r}")
    reference = dissipators.local_loss(int(n_qubits), float(gamma))
    target = dissipators.jump_strength(reference)
    raw = _DISSIPATOR_BUILDERS[family](int(n_qubits), float(gamma))
    raw_strength = dissipators.jump_strength(raw)
    jumps = dissipators.normalize_jump_strength(raw, target)
    matched_strength = dissipators.jump_strength(jumps)
    return jumps, {
        "reference_jump_strength": float(target),
        "raw_jump_strength": float(raw_strength),
        "matched_jump_strength": float(matched_strength),
        "jump_rate_rescale": float(target / raw_strength),
    }


def build_reservoir(
    *,
    n_qubits: int,
    family: str,
    gamma: float,
    coupling_seed: int,
    config: StressConfig,
) -> tuple[reservoirs.LindbladReservoir, dict[str, float]]:
    """Construct a paired teacher using the canonical Hamiltonian convention."""

    rng = np.random.default_rng(int(coupling_seed))
    couplings = reservoirs.random_couplings(
        int(n_qubits), float(config.coupling_scale), rng
    )
    h_static = reservoirs.ising_xx_hamiltonian(
        couplings, float(config.h), int(n_qubits)
    )
    h_drive = float(config.h_in) * reservoirs.transverse_drive(int(n_qubits))
    jumps, budget = normalized_jump_family(family, int(n_qubits), float(gamma))
    model = reservoirs.LindbladReservoir.from_terms(
        int(n_qubits),
        h_static,
        h_drive,
        jumps,
        float(config.dt),
        cache_propagators=True,
        quantize=(
            None if config.quantize_levels is None else int(config.quantize_levels)
        ),
    )
    return model, budget


def make_suffix_panel(
    config: StressConfig,
    root: RootSpec,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Generate nested suffixes with a unique newest symbol in every group."""

    prefix_rng = np.random.default_rng(int(root.prefix_seed))
    suffix_rng = np.random.default_rng(int(root.suffix_seed))
    longest = max(config.windows)
    if config.quantize_levels is None:
        prefixes = prefix_rng.random(
            (
                int(config.suffix_groups),
                int(config.prefix_branches),
                int(config.prefix_burnin),
            )
        )
        longest_suffixes = suffix_rng.random(
            (int(config.suffix_groups), int(longest))
        )
        if len(np.unique(longest_suffixes[:, -1])) != int(config.suffix_groups):
            raise RuntimeError("continuous newest suffix values were not unique")
    else:
        levels = int(config.quantize_levels)
        prefixes = prefix_rng.integers(
            0,
            levels,
            size=(
                int(config.suffix_groups),
                int(config.prefix_branches),
                int(config.prefix_burnin),
            ),
        ).astype(float) / (levels - 1)
        newest = suffix_rng.permutation(levels)[: int(config.suffix_groups)]
        if longest == 1:
            integer_suffixes = newest[:, None]
        else:
            older = suffix_rng.integers(
                0,
                levels,
                size=(int(config.suffix_groups), int(longest) - 1),
            )
            integer_suffixes = np.column_stack([older, newest])
        longest_suffixes = integer_suffixes.astype(float) / (levels - 1)
    suffixes = {
        int(window): longest_suffixes[:, -int(window) :].copy()
        for window in config.windows
    }
    return prefixes, suffixes


def _prefix_states(
    model: reservoirs.LindbladReservoir,
    prefixes: np.ndarray,
) -> np.ndarray:
    groups, branches, _ = prefixes.shape
    dim = 2 ** int(model.n_qubits)
    states = np.empty((groups, branches, dim, dim), dtype=complex)
    for group, branch in product(range(groups), range(branches)):
        state = model.initial_state()
        for value in prefixes[group, branch]:
            state = model.step(state, float(value))
        states[group, branch] = state
    return states


def _step_propagator(
    model: reservoirs.LindbladReservoir,
    value: float,
) -> np.ndarray:
    """Retrieve the same cached one-step transfer matrix used by ``step``."""

    snapped = model._snap(float(value))  # frozen internal API, used read-only
    return model._sub_propagator(snapped, float(model.dt))


def _suffix_product(
    model: reservoirs.LindbladReservoir,
    suffix: Sequence[float],
) -> np.ndarray:
    width = int(model.base_super.shape[0])
    transfer = np.eye(width, dtype=complex)
    for value in suffix:
        transfer = _step_propagator(model, float(value)) @ transfer
    return transfer


def observable_matrix(observables: Sequence[readout.Observable]) -> np.ndarray:
    """Matrix ``C`` satisfying ``C vec(rho) = [Tr(rho O_j)]_j``."""

    if not observables:
        raise ValueError("at least one observable is required")
    return np.stack(
        [np.asarray(item.matrix, complex).T.reshape(-1, order="F") for item in observables]
    )


def _expectations(states: np.ndarray, c_matrix: np.ndarray) -> np.ndarray:
    flat = np.stack([liouvillian.vec(state) for state in states])
    values = flat @ c_matrix.T
    if float(np.max(np.abs(values.imag))) > 2e-9:
        raise RuntimeError("Hermitian-state expectations acquired a large imaginary part")
    return values.real


def _trace_zero_projected_gain(
    c_matrix: np.ndarray,
    transfer: np.ndarray,
    dim: int,
) -> float:
    identity = liouvillian.vec(np.eye(dim, dtype=complex))
    mapped = c_matrix @ transfer
    restricted = mapped - np.outer(mapped @ identity, identity.conj()) / float(dim)
    return float(np.linalg.svd(restricted, compute_uv=False)[0])


def _sample_span_gain(
    c_matrix: np.ndarray,
    transfer: np.ndarray,
    centered_prefix_vectors: np.ndarray,
) -> float:
    """Upper-bound gain on a numerical superset of the sampled input span.

    All left singular vectors returned by the thin SVD are retained, including
    vectors associated with numerically tiny or zero singular values.  Their
    span contains every supplied prefix-difference vector, so the induced norm
    remains a rigorous upper bound.  Retaining extra null-space directions can
    only loosen the bound; dropping small nonzero directions could invalidate
    it.
    """

    matrix = np.asarray(centered_prefix_vectors, complex).T
    if matrix.size == 0 or not np.any(matrix):
        return 0.0
    basis = np.linalg.svd(matrix, full_matrices=False)[0]
    return float(np.linalg.svd(c_matrix @ transfer @ basis, compute_uv=False)[0])


def diagnose_window(
    *,
    model: reservoirs.LindbladReservoir,
    prefix_states: np.ndarray,
    suffixes: np.ndarray,
    observables: Sequence[readout.Observable],
) -> dict[str, float | int]:
    """Measure one window and verify transfer predictions and upper bounds."""

    groups, branches = prefix_states.shape[:2]
    if suffixes.shape[0] != groups:
        raise ValueError("suffix group count must match prefix states")
    if branches < 2:
        raise ValueError("at least two prefix branches are required")
    c_matrix = observable_matrix(observables)
    n_outputs = int(c_matrix.shape[0])
    dim = int(prefix_states.shape[-1])

    direct_outputs: list[np.ndarray] = []
    initial_outputs: list[np.ndarray] = []
    predicted_centered_outputs: list[np.ndarray] = []
    direct_centered_outputs: list[np.ndarray] = []
    initial_state_mses: list[float] = []
    final_state_mses: list[float] = []
    full_bounds: list[float] = []
    span_bounds: list[float] = []
    full_gains: list[float] = []
    span_gains: list[float] = []
    half_diameters: list[float] = []

    for group in range(groups):
        initial = np.asarray(prefix_states[group], complex)
        initial_mean = np.mean(initial, axis=0)
        initial_delta = initial - initial_mean[None, :, :]
        initial_vectors = np.stack([liouvillian.vec(state) for state in initial_delta])
        initial_state_mse = float(np.mean(np.sum(np.abs(initial_vectors) ** 2, axis=1)))

        transfer = _suffix_product(model, suffixes[group])
        predicted_delta_vectors = (transfer @ initial_vectors.T).T
        predicted_outputs = (predicted_delta_vectors @ c_matrix.T).real

        final_states = np.empty_like(initial)
        for branch in range(branches):
            state = initial[branch].copy()
            for value in suffixes[group]:
                state = model.step(state, float(value))
            final_states[branch] = state
        final_mean = np.mean(final_states, axis=0)
        final_delta = final_states - final_mean[None, :, :]
        final_vectors = np.stack([liouvillian.vec(state) for state in final_delta])
        final_state_mse = float(np.mean(np.sum(np.abs(final_vectors) ** 2, axis=1)))

        y_initial = _expectations(initial, c_matrix)
        y_final = _expectations(final_states, c_matrix)
        y_centered = y_final - np.mean(y_final, axis=0, keepdims=True)
        pair_distances = [
            float(np.sqrt(np.mean((y_final[first] - y_final[second]) ** 2)))
            for first in range(branches)
            for second in range(first + 1, branches)
        ]

        full_gain = _trace_zero_projected_gain(c_matrix, transfer, dim)
        span_gain = _sample_span_gain(c_matrix, transfer, initial_vectors)
        initial_state_mses.append(initial_state_mse)
        final_state_mses.append(final_state_mse)
        full_bounds.append(full_gain**2 * initial_state_mse / n_outputs)
        span_bounds.append(span_gain**2 * initial_state_mse / n_outputs)
        full_gains.append(full_gain / np.sqrt(n_outputs))
        span_gains.append(span_gain / np.sqrt(n_outputs))
        half_diameters.append(0.5 * max(pair_distances))
        direct_outputs.append(y_final)
        initial_outputs.append(y_initial)
        predicted_centered_outputs.append(predicted_outputs)
        direct_centered_outputs.append(y_centered)

    output_cube = np.stack(direct_outputs)
    initial_cube = np.stack(initial_outputs)
    direct_centered = np.stack(direct_centered_outputs)
    predicted_centered = np.stack(predicted_centered_outputs)
    history_mse = float(np.mean(direct_centered**2))
    transfer_predicted_mse = float(np.mean(predicted_centered**2))
    initial_centered = initial_cube - np.mean(initial_cube, axis=1, keepdims=True)
    initial_observable_mse = float(np.mean(initial_centered**2))
    global_mean = np.mean(output_cube, axis=(0, 1), keepdims=True)
    target_variance = float(np.mean((output_cube - global_mean) ** 2))
    history_nrmse = (
        float(np.sqrt(history_mse / target_variance))
        if target_variance > 0
        else float("inf")
    )
    prediction_error = float(np.max(np.abs(predicted_centered - direct_centered)))
    full_bound = float(np.mean(full_bounds))
    span_bound = float(np.mean(span_bounds))
    initial_state_mse = float(np.mean(initial_state_mses))
    final_state_mse = float(np.mean(final_state_mses))
    tolerance = 2e-10
    if prediction_error > tolerance:
        raise RuntimeError(
            f"transfer prediction mismatch {prediction_error:.3e} exceeds {tolerance:.1e}"
        )
    if full_bound + tolerance < history_mse or span_bound + tolerance < history_mse:
        raise RuntimeError("observable-restricted transfer upper bound was violated")

    def root_ratio(numerator: float, denominator: float) -> float:
        return float(np.sqrt(numerator / denominator)) if denominator > 0 else 0.0

    return {
        "window": int(suffixes.shape[1]),
        "n_suffix_groups": int(groups),
        "prefix_branches": int(branches),
        "n_outputs": n_outputs,
        "target_variance": target_variance,
        "history_mse": history_mse,
        "history_nrmse": history_nrmse,
        "initial_observable_mse": initial_observable_mse,
        "observable_memory_gain": root_ratio(history_mse, initial_observable_mse),
        "initial_state_hs_mse": initial_state_mse,
        "final_state_hs_mse": final_state_mse,
        "state_hs_memory_gain": root_ratio(final_state_mse, initial_state_mse),
        "transfer_predicted_history_mse": transfer_predicted_mse,
        "max_transfer_prediction_error": prediction_error,
        "tracezero_observable_gain_mean": float(np.mean(full_gains)),
        "tracezero_observable_gain_max": float(np.max(full_gains)),
        "sample_span_observable_gain_mean": float(np.mean(span_gains)),
        "sample_span_observable_gain_max": float(np.max(span_gains)),
        "tracezero_upper_mse": full_bound,
        "sample_span_upper_mse": span_bound,
        "tracezero_bound_slack": float(full_bound - history_mse),
        "sample_span_bound_slack": float(span_bound - history_mse),
        "max_half_output_diameter": float(np.max(half_diameters)),
    }


def _panel_observables(panel: ObservablePanel, n_qubits: int) -> list[readout.Observable]:
    return readout.pauli_observables(
        int(n_qubits), axes=panel.axes, max_weight=int(panel.max_weight)
    )


def run_stress_test(config: StressConfig) -> pd.DataFrame:
    """Run the paired factorial diagnostic and return a sorted science table."""

    validate_config(config)
    rows: list[dict[str, Any]] = []
    for root in config.roots:
        prefixes, suffix_map = make_suffix_panel(config, root)
        for n_qubits in config.n_qubits:
            coupling_seed = int(root.coupling_seeds[int(n_qubits)])
            panel_observables = {
                panel.name: _panel_observables(panel, int(n_qubits))
                for panel in config.observable_panels
            }
            for family, gamma in product(config.dissipator_families, config.gammas):
                model, budget = build_reservoir(
                    n_qubits=int(n_qubits),
                    family=str(family),
                    gamma=float(gamma),
                    coupling_seed=coupling_seed,
                    config=config,
                )
                prefix_states = _prefix_states(model, prefixes)
                for window in config.windows:
                    suffixes = suffix_map[int(window)]
                    for panel in config.observable_panels:
                        diagnostic = diagnose_window(
                            model=model,
                            prefix_states=prefix_states,
                            suffixes=suffixes,
                            observables=panel_observables[panel.name],
                        )
                        rows.append(
                            {
                                "root_id": root.root_id,
                                "coupling_seed": coupling_seed,
                                "prefix_seed": int(root.prefix_seed),
                                "suffix_seed": int(root.suffix_seed),
                                "teacher_n_qubits": int(n_qubits),
                                "dissipator_family": str(family),
                                "reference_local_loss_gamma": float(gamma),
                                "observable_panel": panel.name,
                                "observable_axes": "".join(panel.axes),
                                "observable_max_weight": int(panel.max_weight),
                                **budget,
                                **diagnostic,
                            }
                        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        [
            "root_id",
            "teacher_n_qubits",
            "dissipator_family",
            "reference_local_loss_gamma",
            "observable_panel",
            "window",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def fit_decay_curves(results: pd.DataFrame) -> pd.DataFrame:
    """Fit log measured conditional risk versus window within each root cell."""

    keys = [
        "root_id",
        "teacher_n_qubits",
        "dissipator_family",
        "reference_local_loss_gamma",
        "observable_panel",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in results.groupby(keys, sort=True):
        ordered = group.sort_values("window")
        x = ordered["window"].to_numpy(dtype=float)
        y = ordered["history_nrmse"].to_numpy(dtype=float)
        valid = np.isfinite(y) & (y > 0)
        status = "FIT"
        slope = intercept = r_squared = None
        if int(np.sum(valid)) < 3:
            status = "INSUFFICIENT_POSITIVE_POINTS"
        else:
            design = np.column_stack([np.ones(int(np.sum(valid))), x[valid]])
            coefficients, _, rank, _ = np.linalg.lstsq(
                design, np.log(y[valid]), rcond=None
            )
            if int(rank) != 2:
                status = "RANK_DEFICIENT"
            else:
                fitted = design @ coefficients
                truth = np.log(y[valid])
                total = float(np.sum((truth - np.mean(truth)) ** 2))
                residual = float(np.sum((truth - fitted) ** 2))
                slope = float(coefficients[1])
                intercept = float(coefficients[0])
                r_squared = float(1.0 - residual / total) if total > 0 else 0.0
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "fit_status": status,
                "n_points": int(np.sum(valid)),
                "log_nrmse_intercept": intercept,
                "log_nrmse_slope_per_input": slope,
                "r_squared": r_squared,
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="mergesort").reset_index(drop=True)


def summarize_results(
    results: pd.DataFrame,
    decay_fits: pd.DataFrame,
    config: StressConfig,
) -> dict[str, Any]:
    """Create a strict-JSON-ready, root-aware descriptive summary."""

    group_keys = [
        "teacher_n_qubits",
        "dissipator_family",
        "reference_local_loss_gamma",
        "observable_panel",
        "window",
    ]
    aggregates: list[dict[str, Any]] = []
    for key, group in results.groupby(group_keys, sort=True):
        aggregates.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "n_roots": int(group["root_id"].nunique()),
                "median_history_nrmse": float(np.median(group["history_nrmse"])),
                "minimum_history_nrmse": float(np.min(group["history_nrmse"])),
                "maximum_history_nrmse": float(np.max(group["history_nrmse"])),
                "median_observable_memory_gain": float(
                    np.median(group["observable_memory_gain"])
                ),
                "median_state_hs_memory_gain": float(
                    np.median(group["state_hs_memory_gain"])
                ),
            }
        )
    fit_keys = [
        "teacher_n_qubits",
        "dissipator_family",
        "reference_local_loss_gamma",
        "observable_panel",
    ]
    fit_aggregates: list[dict[str, Any]] = []
    for key, group in decay_fits.groupby(fit_keys, sort=True):
        defined = group[
            group["fit_status"].eq("FIT")
            & group["log_nrmse_slope_per_input"].notna()
            & group["r_squared"].notna()
        ]
        fit_aggregates.append(
            {
                **dict(zip(fit_keys, key, strict=True)),
                "defined_roots": int(len(defined)),
                "negative_slope_roots": int(
                    np.sum(defined["log_nrmse_slope_per_input"] < 0)
                ),
                "median_log_nrmse_slope_per_input": (
                    float(np.median(defined["log_nrmse_slope_per_input"]))
                    if len(defined)
                    else None
                ),
                "median_r_squared": (
                    float(np.median(defined["r_squared"])) if len(defined) else None
                ),
            }
        )
    return {
        "schema_version": 1,
        "study": "pra_observable_memory_extension",
        "status": "EXPLORATORY_STRESS_TEST",
        "claim_boundary": (
            "finite sampled conditional risks and observable-restricted "
            "Hilbert-Schmidt transfer bounds; no uniform contraction theorem"
        ),
        "config": config_to_dict(config),
        "counts": {
            "rows": int(len(results)),
            "decay_fits": int(len(decay_fits)),
            "roots": int(results["root_id"].nunique()),
            "n_qubit_values": sorted(
                int(value) for value in results["teacher_n_qubits"].unique()
            ),
            "dissipator_families": sorted(
                str(value) for value in results["dissipator_family"].unique()
            ),
        },
        "numerical_audit": {
            "max_transfer_prediction_error": float(
                results["max_transfer_prediction_error"].max()
            ),
            "min_tracezero_bound_slack": float(
                results["tracezero_bound_slack"].min()
            ),
            "min_sample_span_bound_slack": float(
                results["sample_span_bound_slack"].min()
            ),
            "jump_strength_max_abs_mismatch": float(
                np.max(
                    np.abs(
                        results["matched_jump_strength"]
                        - results["reference_jump_strength"]
                    )
                )
            ),
        },
        "condition_medians": aggregates,
        "decay_fit_medians": fit_aggregates,
    }


def config_to_dict(config: StressConfig) -> dict[str, Any]:
    """Convert nested immutable dataclasses and mappings to JSON-safe objects."""

    raw = asdict(config)
    for root in raw["roots"]:
        root["coupling_seeds"] = {
            str(key): int(value) for key, value in root["coupling_seeds"].items()
        }
    return raw


def compact_findings(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the most legible d=1 to d=max root-median history contrasts."""

    conditions = list(summary["condition_medians"])
    windows = sorted({int(row["window"]) for row in conditions})
    if not windows:
        return []
    small, large = windows[0], windows[-1]
    keys = [
        "teacher_n_qubits",
        "dissipator_family",
        "reference_local_loss_gamma",
        "observable_panel",
    ]
    indexed = {
        tuple(row[key] for key in keys) + (int(row["window"]),): row
        for row in conditions
    }
    output: list[dict[str, Any]] = []
    base_keys = sorted({key[:-1] for key in indexed})
    for key in base_keys:
        first = indexed.get(key + (small,))
        last = indexed.get(key + (large,))
        if first is None or last is None:
            continue
        initial = float(first["median_history_nrmse"])
        final = float(last["median_history_nrmse"])
        output.append(
            {
                **dict(zip(keys, key, strict=True)),
                "window_contrast": [small, large],
                "median_history_nrmse_d_min": initial,
                "median_history_nrmse_d_max": final,
                "fractional_reduction": (
                    float(1.0 - final / initial) if initial > 0 else None
                ),
            }
        )
    return output
