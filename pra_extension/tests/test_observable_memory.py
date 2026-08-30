from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "pra_extension" / "code"
SRC_ROOT = REPO_ROOT / "src"
for path in (CODE_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from observable_memory import (  # noqa: E402
    _sample_span_gain,
    build_reservoir,
    default_config,
    diagnose_window,
    fit_decay_curves,
    make_suffix_panel,
    normalized_jump_family,
    observable_matrix,
    run_stress_test,
)
from qrc_vs_qelm.core import liouvillian, operators, readout  # noqa: E402
from qrc_vs_qelm.provenance import canonical_material_files  # noqa: E402


def test_pra_extension_is_outside_canonical_material_boundary() -> None:
    files = canonical_material_files(
        repo_root=REPO_ROOT,
        config_path=REPO_ROOT / "configs" / "canonical_emulation.yaml",
        runner_path=REPO_ROOT / "scripts" / "run_canonical_emulation.py",
    )
    relative = {path.relative_to(REPO_ROOT).as_posix() for path in files.values()}
    assert all(not path.startswith("pra_extension/") for path in relative)


def test_jump_families_are_strength_matched() -> None:
    for n_qubits in (2, 3, 4):
        strengths = []
        rescales = {}
        for family in ("local_loss", "dephasing", "collective_loss"):
            _, budget = normalized_jump_family(family, n_qubits, 1.25)
            strengths.append(budget["matched_jump_strength"])
            rescales[family] = budget["jump_rate_rescale"]
        np.testing.assert_allclose(strengths, strengths[0], rtol=1e-13, atol=1e-13)
        assert rescales["local_loss"] == 1.0
        assert rescales["dephasing"] == 0.5
        assert np.isclose(rescales["collective_loss"], 1.0)


def test_observable_matrix_matches_direct_expectations() -> None:
    observables = readout.pauli_observables(2, axes=("x", "z"), max_weight=2)
    c_matrix = observable_matrix(observables)
    rng = np.random.default_rng(991)
    matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = matrix @ matrix.conj().T
    rho = rho / np.trace(rho)
    via_matrix = (c_matrix @ liouvillian.vec(rho)).real
    direct = np.asarray([readout.expectation(rho, item.matrix) for item in observables])
    np.testing.assert_allclose(via_matrix, direct, atol=1e-13, rtol=1e-13)


def test_full_pauli_panel_obeys_parseval_and_subpanels_are_fractions() -> None:
    """Audit the normalization behind the exact visibility fraction."""

    rng = np.random.default_rng(28_082_812)
    for n_qubits in (2, 3):
        dim = 2**n_qubits
        states = []
        for _ in range(7):
            matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(
                size=(dim, dim)
            )
            state = matrix @ matrix.conj().T
            states.append(state / np.trace(state))
        states_array = np.asarray(states)
        centered = states_array - np.mean(states_array, axis=0, keepdims=True)
        state_hs_mse = float(
            np.mean(np.einsum("bij,bij->b", centered.conj(), centered).real)
        )

        full = [
            operators.pauli_string(labels, n_qubits)
            for labels in product("ixyz", repeat=n_qubits)
            if any(label != "i" for label in labels)
        ]
        full_coefficients = np.asarray(
            [
                [readout.expectation(delta, matrix) for matrix in full]
                for delta in centered
            ]
        )
        full_energy = float(np.mean(np.sum(full_coefficients**2, axis=1)))
        np.testing.assert_allclose(
            full_energy, dim * state_hs_mse, atol=2e-12, rtol=2e-12
        )

        local_z = readout.pauli_observables(
            n_qubits, axes=("z",), max_weight=1
        )
        local_z_energy = float(
            np.mean(
                np.sum(
                    np.asarray(
                        [
                            [
                                readout.expectation(delta, item.matrix)
                                for item in local_z
                            ]
                            for delta in centered
                        ]
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
        assert 0.0 <= local_z_energy / full_energy <= 1.0 + 2e-13


def test_transfer_prediction_and_observable_bounds_hold() -> None:
    config = default_config("smoke")
    root = config.roots[0]
    prefixes, suffix_map = make_suffix_panel(config, root)
    model, _ = build_reservoir(
        n_qubits=2,
        family="collective_loss",
        gamma=1.0,
        coupling_seed=root.coupling_seeds[2],
        config=config,
    )
    states = np.empty((config.suffix_groups, config.prefix_branches, 4, 4), complex)
    for group in range(config.suffix_groups):
        for branch in range(config.prefix_branches):
            state = model.initial_state()
            for value in prefixes[group, branch]:
                state = model.step(state, float(value))
            states[group, branch] = state
    observables = readout.pauli_observables(2, axes=("x", "y", "z"), max_weight=2)
    result = diagnose_window(
        model=model,
        prefix_states=states,
        suffixes=suffix_map[2],
        observables=observables,
    )
    assert result["max_transfer_prediction_error"] < 2e-10
    assert result["tracezero_bound_slack"] >= -2e-10
    assert result["sample_span_bound_slack"] >= -2e-10
    np.testing.assert_allclose(
        result["transfer_predicted_history_mse"],
        result["history_mse"],
        atol=2e-10,
        rtol=1e-9,
    )


def test_sample_span_bound_retains_tiny_physical_directions() -> None:
    epsilon = 1e-12
    centered = np.asarray(
        [[1.0, epsilon], [-1.0, epsilon], [0.0, -2.0 * epsilon]],
        dtype=complex,
    )
    c_matrix = np.asarray([[0.0, 1.0]], dtype=complex)
    transfer = np.eye(2, dtype=complex)

    gain = _sample_span_gain(c_matrix, transfer, centered)
    actual_mse = float(np.mean(np.abs(centered @ (c_matrix @ transfer).T) ** 2))
    initial_mse = float(np.mean(np.sum(np.abs(centered) ** 2, axis=1)))

    np.testing.assert_allclose(gain, 1.0, atol=1e-13, rtol=1e-13)
    assert gain**2 * initial_mse >= actual_mse


def test_smoke_factorial_is_deterministic_and_complete() -> None:
    config = replace(default_config("smoke"), prefix_burnin=4)
    first = run_stress_test(config)
    second = run_stress_test(config)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    expected = (
        len(config.roots)
        * len(config.n_qubits)
        * len(config.dissipator_families)
        * len(config.gammas)
        * len(config.windows)
        * len(config.observable_panels)
    )
    assert len(first) == expected
    assert set(first["dissipator_family"]) == {
        "local_loss",
        "dephasing",
        "collective_loss",
    }
    assert float(first["max_transfer_prediction_error"].max()) < 2e-10
    assert float(first["tracezero_bound_slack"].min()) >= -2e-10
    assert float(first["sample_span_bound_slack"].min()) >= -2e-10
    fits = fit_decay_curves(first)
    expected_fits = (
        len(config.roots)
        * len(config.n_qubits)
        * len(config.dissipator_families)
        * len(config.gammas)
        * len(config.observable_panels)
    )
    assert len(fits) == expected_fits
    assert set(fits["fit_status"]) == {"INSUFFICIENT_POSITIVE_POINTS"}


def test_continuous_profile_has_nested_off_grid_inputs() -> None:
    config = default_config("continuous")
    assert config.quantize_levels is None
    assert config.n_qubits == (2, 3)
    assert len(config.roots) == 12
    prefixes, suffixes = make_suffix_panel(config, config.roots[0])
    assert np.all((prefixes >= 0.0) & (prefixes <= 1.0))
    assert np.all((suffixes[8] >= 0.0) & (suffixes[8] <= 1.0))
    np.testing.assert_array_equal(suffixes[1], suffixes[8][:, -1:])
    np.testing.assert_array_equal(suffixes[4], suffixes[8][:, -4:])
    # These values are generated directly in [0,1), not snapped to the
    # 13-symbol grid used by the main exploratory factorial.
    assert np.any(np.abs(prefixes * 12.0 - np.rint(prefixes * 12.0)) > 1e-8)
    assert np.any(np.abs(suffixes[8] * 12.0 - np.rint(suffixes[8] * 12.0)) > 1e-8)
