"""Check actual schedules, original budgets, fixed input access and provenance."""

import json
import numpy as np
import pandas as pd
import pytest

import run_feature_density as study


@pytest.mark.parametrize("m", [1, 2, 3, 4, 6, 8])
def test_uniform_schedule_matches_direct_evolution(m):
    p = study.base._canonical_profile(
        {"max_snapshots": 24, "input_levels": 33}, {"student_n_qubits": 2}
    )
    fine = study.efficient_student_channel(p, 12)
    obs = study.base.canonical._student_observables(p)
    windows = np.array([[0.1, 0.8], [0.8, 0.1], [0, 1]])
    full = fine.features(windows, obs)
    p["max_snapshots"] = m
    direct = study.efficient_student_channel(p, 12).features(windows, obs)
    np.testing.assert_allclose(
        study.subset(full, m, len(obs)), direct, atol=2e-13, rtol=2e-13
    )


def test_original_nested_banks_preserved():
    full = np.arange(2 * 24 * 18).reshape(2, -1)
    for m in (1, 2, 4):
        expected = study.base.select_nested_snapshots(
            study.subset(full, 4, 18),
            n_observables=18,
            max_snapshots=4,
            selected_snapshots=m,
        )
        np.testing.assert_array_equal(study.subset(full, m, 18), expected)


def test_bad_schedule_or_shape_fails():
    with pytest.raises(ValueError):
        study.subset(np.ones((2, 432)), 5, 18)
    with pytest.raises(ValueError):
        study.subset(np.ones((2, 72)), 4, 18)


def test_deterministic_smokes_and_fixed_panel(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    study.run(a, smoke=True)
    study.run(b, smoke=True)
    ma = json.loads((a / "manifest.json").read_text())
    mb = json.loads((b / "manifest.json").read_text())
    assert ma["scientific_sha256"] == mb["scientific_sha256"]
    f = pd.read_csv(a / "metrics.csv")
    assert len(f) == 7 and f.target_variance.nunique() == 1
    assert f.history_mse.nunique() == 1
    assert (f.fit_rows == f.train_rows + f.val_rows).all()
    np.testing.assert_allclose(
        f.total_mse, f.history_mse + f.representation_mse, atol=1e-13
    )
    np.testing.assert_allclose(f.total_nrmse, np.sqrt(f.total_mse / f.target_variance))
    with pytest.raises(FileExistsError):
        study.run(a, smoke=True)
    with pytest.raises(ValueError):
        study.coverage(pd.concat([f, f.iloc[:1]]), True)
    with pytest.raises(ValueError):
        study.coverage(f.iloc[:-1], True)


def test_development_and_production_seeds_disjoint():
    cfg = study.read_config()
    prod = study.ledger(cfg)
    dev = study.ledger(cfg, True)
    assert len(prod) == 12 * 3 * 7 and len(dev) == 7
    values = [e["seed"] for e in prod + dev]
    assert len(set(values)) == len(values)


def test_material_changes_reject_freeze(tmp_path):
    freeze = tmp_path / "freeze.json"
    study.dump(freeze, {"status": "FROZEN", "material": study.material()})
    study.verify(freeze=freeze)
    study.dump(freeze, {"status": "FROZEN", "material": {}})
    with pytest.raises(ValueError):
        study.verify(freeze=freeze)
