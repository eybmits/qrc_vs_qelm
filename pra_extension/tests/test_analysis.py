from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from analyze_pra_extension import (
    analyze_observable,
    bootstrap_median,
    derive_measurement_visibility,
    select_budget_on_validation,
    validate_observable_v2_profile,
    validate_shot_v2_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OBSERVABLE_V2 = (
    REPO_ROOT
    / "pra_extension"
    / "outputs"
    / "observable_memory_full_exploratory_v2_12roots"
    / "observable_memory_results.csv"
)
SHOTS_V2 = (
    REPO_ROOT
    / "pra_extension"
    / "outputs"
    / "finite_shots_full_exploratory_v2_12roots"
    / "finite_shot_results.csv"
)


@pytest.fixture(scope="module")
def observable_v2() -> pd.DataFrame:
    return pd.read_csv(OBSERVABLE_V2)


@pytest.fixture(scope="module")
def shots_v2() -> pd.DataFrame:
    return pd.read_csv(SHOTS_V2)


def test_bootstrap_median_is_root_counted_and_deterministic() -> None:
    first = bootstrap_median([0.1, 0.2, 0.3], seed=7)
    second = bootstrap_median([0.1, 0.2, 0.3], seed=7)
    assert first == second
    assert first["n_roots"] == 3
    assert first["positive_roots"] == 3
    assert first["median"] == 0.2


def test_budget_selection_never_uses_test_error() -> None:
    frame = pd.DataFrame(
        {
            "root_id": ["r1", "r1", "r2", "r2"],
            "teacher_n_qubits": [2, 2, 2, 2],
            "allocation_mode": ["optimal"] * 4,
            "total_shots": [300] * 4,
            "snapshot_budget": [1, 4, 1, 4],
            "validation_expected_objective_mse": [0.1, 0.2, 0.4, 0.3],
            # Test ranking is deliberately reversed.
            "test_expected_total_nrmse": [9.0, 0.1, 0.1, 9.0],
        }
    )
    selected = select_budget_on_validation(frame)
    assert selected["snapshot_budget"].tolist() == [1, 4]
    assert selected["test_expected_total_nrmse"].tolist() == [9.0, 9.0]


def test_observable_v2_exact_profile_and_liveness_contract(
    observable_v2: pd.DataFrame,
) -> None:
    validate_observable_v2_profile(observable_v2)
    _, summary = analyze_observable(observable_v2)
    liveness = {
        row["dissipator_family"]: row
        for row in summary["liveness"]["by_dissipator"]
    }
    assert liveness["dephasing"]["live_rows"] == 413
    assert liveness["dephasing"]["total_rows"] == 1_296
    assert (
        summary["root_block_effects"]["dephasing"]["paired_live_cells_total"]
        == 79
    )
    assert summary["root_block_effects"]["local_loss"][
        "paired_live_cells_total"
    ] == 324
    common = summary["common_live_support_sensitivity"]["effects"]
    assert {
        family: effect["common_live_cells_total"]
        for family, effect in common.items()
    } == {
        "collective_loss": 79,
        "dephasing": 79,
        "local_loss": 79,
    }
    assert all(
        effect["fractional_reduction"]["positive_roots"] == 12
        for effect in common.values()
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("reference_local_loss_gamma", 0.3),
        ("root_id", "pra99"),
        ("history_mse", np.inf),
    ],
)
def test_observable_v2_rejects_axis_and_numeric_mutations(
    observable_v2: pd.DataFrame,
    column: str,
    value: object,
) -> None:
    mutated = observable_v2.copy()
    mutated.loc[0, column] = value
    with pytest.raises((ValueError, RuntimeError)):
        validate_observable_v2_profile(mutated)


def test_observable_v2_rejects_duplicate_cartesian_key(
    observable_v2: pd.DataFrame,
) -> None:
    mutated = pd.concat([observable_v2, observable_v2.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate Cartesian key"):
        validate_observable_v2_profile(mutated)


def test_measurement_visibility_matches_parseval_formula_and_interval(
    observable_v2: pd.DataFrame,
) -> None:
    visibility = derive_measurement_visibility(observable_v2)
    expected = (
        observable_v2["n_outputs"]
        * observable_v2["history_mse"]
        / (
            np.power(2.0, observable_v2["teacher_n_qubits"])
            * observable_v2["final_state_hs_mse"]
        )
    )
    np.testing.assert_allclose(visibility["f_vis"], expected, rtol=0.0, atol=0.0)
    assert float(visibility["f_vis"].min()) >= 0.0
    assert float(visibility["f_vis"].max()) <= 1.0

    mutated = observable_v2.copy()
    index = int(visibility["f_vis"].idxmax())
    mutated.loc[index, "final_state_hs_mse"] *= 0.5
    with pytest.raises(RuntimeError, match="Pauli-Parseval interval"):
        derive_measurement_visibility(mutated)


def test_shot_v2_exact_profile(shots_v2: pd.DataFrame) -> None:
    validate_shot_v2_profile(shots_v2)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("feature_budget", 999),
        ("snapshot_indices_zero_based", "[0]"),
        ("test_expected_total_nrmse", np.inf),
    ],
)
def test_shot_v2_rejects_metadata_and_numeric_mutations(
    shots_v2: pd.DataFrame,
    column: str,
    value: object,
) -> None:
    mutated = shots_v2.copy()
    mutated.loc[0, column] = value
    with pytest.raises((ValueError, RuntimeError)):
        validate_shot_v2_profile(mutated)


def test_shot_v2_rejects_nonpositive_or_nonpreserving_allocation(
    shots_v2: pd.DataFrame,
) -> None:
    mutated = shots_v2.copy()
    index = int(mutated[mutated["allocation_mode"].eq("equal")].index[0])
    allocation = [0, 100, 200]
    mutated.loc[index, "allocation_json"] = "[0, 100, 200]"
    mutated.loc[index, "minimum_setting_shots"] = min(allocation)
    mutated.loc[index, "maximum_setting_shots"] = max(allocation)
    with pytest.raises(ValueError, match="positive and budget preserving"):
        validate_shot_v2_profile(mutated)


def test_shot_v2_rejects_duplicate_cartesian_key(
    shots_v2: pd.DataFrame,
) -> None:
    mutated = pd.concat([shots_v2, shots_v2.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate Cartesian keys"):
        validate_shot_v2_profile(mutated)
