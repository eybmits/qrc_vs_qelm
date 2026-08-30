from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "pra_extension" / "code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from finite_shots import (  # noqa: E402
    default_shot_config,
    equal_allocation,
    exact_single_site_features,
    expected_shot_penalty,
    joint_setting_probabilities,
    monte_carlo_entropy_tuple,
    monte_carlo_seed_manifest,
    optimal_allocation,
    outcome_table,
    qelm_statevectors,
    run_shot_study,
    select_nested_measurement_data,
    select_shot_aware_readout,
    setting_contributions,
    setting_covariances,
    validate_shot_config,
)
from qrc_vs_qelm.core import readout  # noqa: E402
from qrc_vs_qelm.core import reservoirs  # noqa: E402
from qrc_vs_qelm.models import qelm  # noqa: E402


def _small_states() -> tuple[np.ndarray, np.ndarray]:
    n = 2
    couplings = np.asarray([[0.0, 0.4], [0.4, 0.0]])
    hamiltonian = reservoirs.ising_xx_hamiltonian(couplings, 0.5, n)
    channel = qelm.UnitaryQELMChannel.from_hamiltonian(
        n_qubits=n, H_static=hamiltonian, T_evolve=0.5, n_snapshots=2
    )
    windows = np.asarray([[0.0, 0.25], [0.5, 0.75], [1.0, 0.5]])
    states = qelm_statevectors(channel, windows)
    return states, exact_single_site_features(states, n)


def test_setting_covariance_is_symmetric_psd_and_has_bernoulli_diagonal() -> None:
    states, features = _small_states()
    covariance = setting_covariances(states, 2)
    assert covariance.shape == (3, 2, 3, 2, 2)
    for row in range(3):
        for snapshot in range(2):
            for axis in range(3):
                matrix = covariance[row, snapshot, axis]
                np.testing.assert_allclose(matrix, matrix.T, atol=1e-13)
                assert float(np.min(np.linalg.eigvalsh(matrix))) >= -1e-12
                start = (snapshot * 3 + axis) * 2
                means = features[row, start : start + 2]
                np.testing.assert_allclose(
                    np.diag(matrix), 1.0 - means**2, atol=1e-12
                )


def test_joint_probabilities_reproduce_single_site_means() -> None:
    states, features = _small_states()
    outcomes = outcome_table(2)
    for snapshot in range(2):
        for axis_index, axis in enumerate(("x", "y", "z")):
            probabilities = joint_setting_probabilities(
                states[1, snapshot], axis, 2
            )
            assert np.all(probabilities >= 0)
            np.testing.assert_allclose(np.sum(probabilities), 1.0, atol=1e-13)
            means = probabilities @ outcomes
            start = (snapshot * 3 + axis_index) * 2
            np.testing.assert_allclose(
                means, features[1, start : start + 2], atol=1e-12
            )


def test_statevector_snapshots_follow_channel_time_order() -> None:
    n = 2
    couplings = np.asarray([[0.0, 0.4], [0.4, 0.0]])
    hamiltonian = reservoirs.ising_xx_hamiltonian(couplings, 0.5, n)
    channel = qelm.UnitaryQELMChannel.from_hamiltonian(
        n_qubits=n, H_static=hamiltonian, T_evolve=0.5, n_snapshots=4
    )
    windows = np.asarray([[0.0, 0.25], [0.5, 0.75]])
    states = qelm_statevectors(channel, windows)
    initial = qelm.encode_window_statevector(windows[0], n, scheme="one_to_one")
    np.testing.assert_allclose(states[0, 0], channel.unitaries[0] @ initial)
    np.testing.assert_allclose(states[0, 3], channel.unitaries[3] @ initial)
    observables = readout.pauli_observables(n, axes=("x", "y", "z"), max_weight=1)
    np.testing.assert_allclose(
        exact_single_site_features(states, n),
        channel.features(windows, observables, encoding_scheme="one_to_one"),
        atol=1e-13,
    )


def test_nested_measurement_selection_uses_T_compatible_snapshots() -> None:
    rows, snapshots, n = 2, 4, 2
    states = np.arange(rows * snapshots * 4).reshape(rows, snapshots, 4)
    features = np.arange(rows * snapshots * 3 * n).reshape(
        rows, snapshots * 3 * n
    )
    covariances = np.arange(rows * snapshots * 3 * n * n).reshape(
        rows, snapshots, 3, n, n
    )
    expected = {1: [3], 2: [1, 3], 4: [0, 1, 2, 3]}
    for budget, indices in expected.items():
        active_states, active_features, active_covariances, selected = (
            select_nested_measurement_data(
                states=states,
                features=features,
                covariances=covariances,
                max_snapshots=snapshots,
                selected_snapshots=budget,
                n_qubits=n,
            )
        )
        np.testing.assert_array_equal(selected, indices)
        np.testing.assert_array_equal(active_states, states[:, indices])
        np.testing.assert_array_equal(active_covariances, covariances[:, indices])
        blocks = features.reshape(rows, snapshots, 3 * n)
        np.testing.assert_array_equal(
            active_features, blocks[:, indices].reshape(rows, -1)
        )


def test_alpha_tie_uses_declared_grid_order_and_frozen_selection_allocation() -> None:
    features = np.zeros((6, 6), float)
    targets = np.zeros((6, 1), float)
    covariances = np.tile(np.eye(2), (6, 1, 3, 1, 1))
    model = select_shot_aware_readout(
        features=features,
        targets=targets,
        covariances=covariances,
        train_idx=np.asarray([0, 1, 2]),
        validation_idx=np.asarray([3, 4]),
        alphas=(1.0, 0.0),
        snapshot_budget=1,
        n_qubits=2,
        allocation_mode="optimal",
        total_shots=12,
    )
    assert model.alpha == 1.0
    np.testing.assert_array_equal(model.allocation, [10, 1, 1])
    assert model.validation_objective == 0.0


def test_deployed_allocation_is_refit_design_optimum_and_ignores_test_rows() -> None:
    rng = np.random.default_rng(47)
    features = rng.normal(size=(10, 6))
    targets = rng.normal(size=(10, 2))
    covariances = np.tile(np.eye(2), (10, 1, 3, 1, 1))
    train = np.asarray([0, 1, 2, 3, 4])
    validation = np.asarray([5, 6, 7])
    kwargs = dict(
        features=features,
        targets=targets,
        train_idx=train,
        validation_idx=validation,
        alphas=(0.0, 1e-3, 1.0),
        snapshot_budget=1,
        n_qubits=2,
        allocation_mode="optimal",
        total_shots=37,
    )
    model = select_shot_aware_readout(covariances=covariances, **kwargs)
    design = np.concatenate([train, validation])
    contributions = setting_contributions(
        covariances[design], model.raw_weights, snapshot_budget=1, n_qubits=2
    )
    expected = optimal_allocation(contributions, 37)
    np.testing.assert_array_equal(model.allocation, expected)
    assert model.design_expected_penalty == pytest.approx(
        expected_shot_penalty(contributions, expected)
    )

    # Held-out feature-state covariances do not enter the deployed design rule.
    mutated = covariances.copy()
    mutated[8:] *= 10_000.0
    rerun = select_shot_aware_readout(covariances=mutated, **kwargs)
    np.testing.assert_array_equal(rerun.allocation, model.allocation)


def test_optimal_integer_allocation_never_worse_than_equal() -> None:
    contributions = np.asarray([4.0, 1.0, 0.25, 2.0])
    equal = equal_allocation(120, len(contributions))
    optimal = optimal_allocation(contributions, 120)
    assert int(np.sum(equal)) == 120
    assert int(np.sum(optimal)) == 120
    assert np.all(optimal >= 1)
    assert expected_shot_penalty(contributions, optimal) <= expected_shot_penalty(
        contributions, equal
    )


def test_monte_carlo_seed_manifest_is_globally_unique_and_deterministic() -> None:
    config = default_shot_config("full")
    first = monte_carlo_seed_manifest(config)
    second = monte_carlo_seed_manifest(config)
    expected = (
        len(config.roots)
        * len(config.teacher_n_qubits)
        * len(config.snapshot_budgets)
        * len(config.total_shots)
        * 2
    )
    assert first == second
    assert len(first) == expected
    assert len({row["entropy"] for row in first}) == expected
    assert len({row["state_uint32"] for row in first}) == expected

    roots = {root.root_id: root for root in config.roots}
    formerly_colliding = (
        monte_carlo_entropy_tuple(
            root=roots["shot04"],
            teacher_n_qubits=2,
            snapshot_budget=2,
            total_shots=300,
            allocation_mode="equal",
        ),
        monte_carlo_entropy_tuple(
            root=roots["shot14"],
            teacher_n_qubits=2,
            snapshot_budget=1,
            total_shots=1_200,
            allocation_mode="equal",
        ),
    )
    assert formerly_colliding[0] != formerly_colliding[1]


def test_n4_profile_has_fresh_complete_unique_seed_ledger() -> None:
    config = default_shot_config("n4")
    assert config.teacher_n_qubits == (4,)
    assert len(config.roots) == 12
    manifest = monte_carlo_seed_manifest(config)
    assert len(manifest) == 12 * 3 * 4 * 2
    assert len({row["entropy"] for row in manifest}) == len(manifest)
    assert len({row["state_uint32"] for row in manifest}) == len(manifest)
    assert all(set(root.teacher_coupling_seeds) == {4} for root in config.roots)


def test_shot_config_rejects_globally_reused_explicit_seed() -> None:
    config = default_shot_config("full")
    duplicated = replace(
        config.roots[1], monte_carlo_seed=config.roots[0].monte_carlo_seed
    )
    invalid = replace(config, roots=(config.roots[0], duplicated, *config.roots[2:]))

    with pytest.raises(ValueError, match="globally unique"):
        validate_shot_config(invalid)


def test_setting_contribution_matches_covariance_quadratic_form() -> None:
    states, _ = _small_states()
    covariance = setting_covariances(states, 2)
    weights = np.arange(12 * 3, dtype=float).reshape(12, 3) / 50.0
    contributions = setting_contributions(
        covariance, weights, snapshot_budget=2, n_qubits=2
    )
    manual = []
    for snapshot in range(2):
        for axis in range(3):
            setting = snapshot * 3 + axis
            block = weights[setting * 2 : setting * 2 + 2]
            values = [
                np.trace(block.T @ covariance[row, snapshot, axis] @ block) / 3
                for row in range(len(states))
            ]
            manual.append(np.mean(values))
    np.testing.assert_allclose(contributions, manual, atol=1e-13, rtol=1e-13)


def test_smoke_shot_study_has_operational_frontier_and_is_deterministic() -> None:
    config = replace(
        default_shot_config("smoke"),
        train_rows=20,
        validation_rows=10,
        test_rows=12,
        burnin=6,
        monte_carlo_repeats=2,
    )
    first = run_shot_study(config)
    second = run_shot_study(config)
    expected_per_budget = 1 + 2 * len(config.total_shots)
    assert len(first) == len(config.snapshot_budgets) * expected_per_budget
    assert set(first["feature_budget"]) == {18, 36, 72}
    assert set(first["allocation_mode"]) == {"noiseless", "equal", "optimal"}
    assert np.all(
        first.loc[first["allocation_mode"] != "noiseless", "test_expected_total_mse"]
        >= first.loc[first["allocation_mode"] != "noiseless", "test_noiseless_mse"]
    )
    # Exact repeatability includes the joint multinomial Monte Carlo audit.
    import pandas as pd

    pd.testing.assert_frame_equal(first, second, check_exact=True)
