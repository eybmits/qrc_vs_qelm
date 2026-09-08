"""Intermediate budgets are real nested, balanced banks with paired controls."""

import json

import numpy as np
import pandas as pd
import pytest

import run_feature_bridge as study


@pytest.mark.parametrize("root", [f"feature{i:02d}" for i in range(1, 13)])
def test_nested_balanced_banks_and_parent_anchors(root):
    full = np.arange(3 * 432).reshape(3, 432)
    previous = set()
    for budget in (18, 24, 30, 36):
        cols = study.columns(budget, root)
        assert len(cols) == len(set(cols)) == budget
        assert previous <= set(cols)
        previous = set(cols)
        np.testing.assert_array_equal(study.subset(full, budget, root), full[:, cols])
        middle = cols[cols < 414] - 198
        assert np.bincount(middle % 6, minlength=6).tolist() == [(budget - 18) // 6] * 6
        assert (
            np.bincount(middle // 6, minlength=3).tolist() == [(budget - 18) // 3] * 3
        )
    for budget in (18, 36):
        np.testing.assert_array_equal(
            study.subset(full, budget, root),
            study.parent.subset(full, budget // 18, 18),
        )


def test_phase_assignment_and_backend_observable_order():
    assert [study.phase(f"feature{i:02d}") for i in range(1, 13)] == [0, 1, 2] * 4
    profile = study.base._canonical_profile(
        {"max_snapshots": 24, "input_levels": 33}, {"student_n_qubits": 6}
    )
    observables = study.base.canonical._student_observables(profile)
    assert len(observables) == 18
    # Test the concrete operator ordering, not merely the selection arithmetic.
    assert [o.name for o in observables] == [
        f"{axis}{i}" for axis in "xyz" for i in range(6)
    ]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        study.columns(25, "feature01")
    with pytest.raises(ValueError):
        study.columns(24, "best_test_root")
    with pytest.raises(ValueError):
        study.subset(np.ones((3, 36)), 24, "feature01")


def test_development_reproduces_parent_and_is_deterministic(tmp_path):
    directories = [tmp_path / "a", tmp_path / "b"]
    for directory in directories:
        study.run(directory, smoke=True)
    ma, mb = [json.loads((d / "manifest.json").read_text()) for d in directories]
    assert ma["scientific_sha256"] == mb["scientific_sha256"]
    assert ma["anchors"]["passed"] and ma["anchors"]["anchor_rows"] == 2
    frame = pd.read_csv(directories[0] / "metrics.csv", float_precision="round_trip")
    assert frame.feature_budget.tolist() == [18, 24, 30, 36]
    assert (frame.fit_rows == frame.train_rows + frame.val_rows).all()
    with pytest.raises(FileExistsError):
        study.run(directories[0], smoke=True)
    with pytest.raises(ValueError):
        study.coverage(frame.iloc[:-1], True)
    with pytest.raises(ValueError):
        study.coverage(pd.concat([frame, frame.iloc[:1]]), True)
    corrupted = frame.copy()
    corrupted.loc[0, "total_mse"] += 0.01
    with pytest.raises(ValueError, match="anchor metric"):
        study.validate_anchors(corrupted, ma["cells"], True)
    with pytest.raises(ValueError, match="scientific arrays"):
        study.validate_anchors(frame, [], True)


def test_config_and_parent_source_still_frozen():
    assert study.read_config()["new_budgets"] == [24, 30]
    study.parent.verify()
    frame, manifest = study.parent_artifacts()
    assert len(frame) == 252 and not manifest["smoke"]
