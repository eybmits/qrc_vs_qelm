from __future__ import annotations

import json

import numpy as np
import pandas as pd

from finite_shots import default_shot_config
from finite_shots_eiv import (
    default_eiv_config,
    eiv_seed_manifest,
    joint_pauli_probabilities,
    run_eiv_study,
    sample_joint_pauli_features,
    select_noisy_readout,
    summarize_eiv_results,
)


def test_joint_sampler_preserves_same_setting_correlations() -> None:
    bell = np.zeros((1, 1, 4), dtype=complex)
    bell[0, 0, 0] = 1.0 / np.sqrt(2.0)
    bell[0, 0, 3] = 1.0 / np.sqrt(2.0)
    probabilities = joint_pauli_probabilities(bell)
    first, allocation = sample_joint_pauli_features(
        probabilities,
        total_shots=3_000,
        rng=np.random.default_rng(17),
    )
    second, _ = sample_joint_pauli_features(
        probabilities,
        total_shots=3_000,
        rng=np.random.default_rng(17),
    )
    np.testing.assert_array_equal(first, second)
    assert allocation.tolist() == [1_000, 1_000, 1_000]
    # Bell-state X and Z outcomes are perfectly correlated shot by shot.  This
    # exact equality would generally fail if qubits were sampled independently.
    assert first[0, 0] == first[0, 1]
    assert first[0, 4] == first[0, 5]


def test_noisy_alpha_selection_uses_validation_trace() -> None:
    rng = np.random.default_rng(18)
    train_x = rng.normal(size=(24, 4))
    validation_x = rng.normal(size=(12, 4))
    coefficient = np.asarray([[1.0], [-0.5], [0.25], [0.0]])
    train_y = train_x @ coefficient
    validation_y = validation_x @ coefficient
    model = select_noisy_readout(
        train_features=train_x,
        train_targets=train_y,
        validation_features=validation_x,
        validation_targets=validation_y,
        alphas=(0.0, 1e-2, 1.0, float("inf")),
    )
    assert model.alpha_index == int(np.argmin(model.validation_alpha_mse))
    assert model.validation_mse == model.validation_alpha_mse[model.alpha_index]


def test_eiv_full_seed_ledger_is_complete_unique_and_fresh() -> None:
    config = default_eiv_config("full")
    previous = default_shot_config("n4")
    ledger = eiv_seed_manifest(config)
    expected = 12 * 3 * 4 * 16 * 3
    assert len(ledger) == expected
    assert len({row["entropy"] for row in ledger}) == expected
    assert len({row["state_uint32"] for row in ledger}) == expected
    assert {root.root_id for root in config.roots}.isdisjoint(
        {f"shotn4_{index:02d}" for index in range(1, 13)}
    )
    new_explicit = {
        seed
        for root in config.roots
        for seed in (
            *root.teacher_coupling_seeds.values(),
            root.student_coupling_seed,
            root.input_seed,
            root.monte_carlo_seed,
        )
    }
    old_explicit = {
        seed
        for root in previous.roots
        for seed in (
            *root.teacher_coupling_seeds.values(),
            root.student_coupling_seed,
            root.input_seed,
            root.monte_carlo_seed,
        )
    }
    assert new_explicit.isdisjoint(old_explicit)


def test_smoke_eiv_is_deterministic_and_selects_budget_on_validation() -> None:
    config = default_eiv_config("smoke")
    first = run_eiv_study(config)
    second = run_eiv_study(config)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert len(first) == 1 * 3 * 2 * 2
    keys = ["root_id", "total_shots", "replicate"]
    assert bool(first.groupby(keys)["selected_budget_on_validation"].sum().eq(1).all())
    for _, group in first.groupby(keys):
        expected = group.sort_values(
            ["validation_mse", "snapshot_budget"], kind="mergesort"
        ).iloc[0]
        selected = group[group["selected_budget_on_validation"]].iloc[0]
        assert int(selected["snapshot_budget"]) == int(expected["snapshot_budget"])
    assert set(first["feature_budget"]) == {18, 36, 72}
    assert set(first["allocation_mode"]) == {"equal_fixed_total"}
    assert first["status"].eq("EXPLORATORY_POST_CONFIRMATORY").all()
    assert summarize_eiv_results(first)["inference_unit"] == (
        "root; Monte Carlo replicates are reduced to a median within root before "
        "root-block uncertainty summaries"
    )
    for row in first.itertuples(index=False):
        trace = np.asarray(json.loads(row.validation_alpha_mse_json), float)
        assert int(row.selected_alpha_index) == int(np.argmin(trace))
        assert json.loads(row.train_entropy)[-1] == 1
        assert json.loads(row.validation_entropy)[-1] == 2
        assert json.loads(row.test_entropy)[-1] == 3
