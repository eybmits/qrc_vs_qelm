"""Small engineering fixtures for the follow-up; never open production roots."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_training_data_control as runner
from qrc_vs_qelm import canonical_study as canonical


@pytest.fixture
def config():
    return {
        "study": "unit_test_training_curve", "training_rows": [480, 960, 1920, 3840],
        "validation_rows": 512, "test_rows": 2048, "reference_rows": 1024,
        "panel_groups": 32, "prefix_branches": 4, "prefix_length": 128,
        "reference_prefix_lengths": [128, 256], "input_levels": 33, "max_snapshots": 4,
        "alpha_grid": [0, 1e-4, 1, "+inf"], "development_seed": 919010,
        "analysis_seed": 919012,
        "anchors": [{"id": "n4_fast", "code": 1, "teacher_n_qubits": 4,
                     "student_n_qubits": 6, "gamma": 4., "window": 6},
                    {"id": "n5_long", "code": 2, "teacher_n_qubits": 5,
                     "student_n_qubits": 10, "gamma": 1., "window": 10}],
        "roots": [{"root_id": "unopened01", "seed": 919011}],
    }


def write_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("mutation", ["root_collision", "analysis_collision", "development_collision",
    "duplicate_anchor", "duplicate_code", "nonmonotone_rows", "negative_rows", "window_too_large",
    "bad_snapshot", "unsafe_root", "bad_alpha", "bad_prefix_lengths"])
def test_config_rejects_invalid_contract(config, tmp_path, mutation):
    cfg = deepcopy(config)
    if mutation == "root_collision":
        cfg["roots"].append({"root_id": "other", "seed": cfg["roots"][0]["seed"]})
    elif mutation == "analysis_collision":
        cfg["analysis_seed"] = cfg["roots"][0]["seed"]
    elif mutation == "development_collision":
        cfg["development_seed"] = cfg["roots"][0]["seed"]
    elif mutation == "duplicate_anchor":
        cfg["anchors"][1]["id"] = cfg["anchors"][0]["id"]
    elif mutation == "duplicate_code":
        cfg["anchors"][1]["code"] = 1
    elif mutation == "nonmonotone_rows":
        cfg["training_rows"] = [960, 480]
    elif mutation == "negative_rows":
        cfg["test_rows"] = -1
    elif mutation == "window_too_large":
        cfg["anchors"][0]["window"] = 7
    elif mutation == "bad_snapshot":
        cfg["max_snapshots"] = 3
    elif mutation == "unsafe_root":
        cfg["roots"][0]["root_id"] = "../old_results"
    elif mutation == "bad_alpha":
        cfg["alpha_grid"] = [-1]
    else:
        cfg["reference_prefix_lengths"] = [64, 256]
    with pytest.raises(ValueError):
        runner.read_config(write_config(tmp_path, cfg))


def test_json_duplicates_and_nonfinite_fail_closed(tmp_path):
    path = tmp_path / "config.json"
    for text in ('{"study":"a","study":"b"}', '{"study":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            runner.read_config(path)


def test_profiles_and_seed_roles_never_open_production(config):
    original = deepcopy(config)
    smoke = runner.profile_config(config, "smoke")
    benchmark = runner.profile_config(config, "benchmark")
    assert config == original
    assert smoke["roots"][0]["root_id"] == "development_smoke"
    assert benchmark["roots"][0]["root_id"] == "development_benchmark"
    assert smoke["training_rows"] == [8, 16, 32, 64]
    assert benchmark["anchors"][0]["student_n_qubits"] == 10
    assert benchmark["prefix_length"] == 128
    ledger = runner.seed_manifest(config)
    assert len(ledger) == (2 + 1 + 1) * len(runner.SEED_ROLES)
    assert len({tuple(item["entropy"]) for item in ledger}) == len(ledger)
    assert len({item["seed"] for item in ledger}) == len(ledger)
    assert runner.seed_manifest(config) == ledger
    development = {item["seed"] for item in ledger if item["profile"] != "production"}
    production = {item["seed"] for item in ledger if item["profile"] == "production"}
    assert development.isdisjoint(production)


def test_fitting_scaler_and_selection_ignore_all_evaluation_data():
    rng = np.random.default_rng(34)
    features = {name: rng.normal(size=(count, 3)) for name, count in
                (("train", 40), ("val", 12), ("test", 9), ("reference", 8), ("panel", 4))}
    targets = {name: value @ np.array([[1., 0.], [.3, .2], [0., .5]])
               for name, value in features.items() if name in {"train", "val", "test"}}
    features["train"][20:] += 1e4
    first = runner.fit_curve_row(features, targets, 20, (0., 1., float("inf")))
    np.testing.assert_array_equal(first.feature_mean, features["train"][:20].mean(axis=0))
    np.testing.assert_array_equal(first.feature_scale, features["train"][:20].std(axis=0))
    changed, changed_targets = deepcopy(features), deepcopy(targets)
    for key in ("test", "reference", "panel"):
        changed[key] *= 1e12
    changed["train"][20:] *= 1e6
    changed_targets["test"] *= -1e9
    changed_targets["train"][20:] *= -1e9
    second = runner.fit_curve_row(changed, changed_targets, 20, (0., 1., float("inf")))
    for attribute in ("weights", "intercept", "feature_mean", "feature_scale"):
        np.testing.assert_array_equal(getattr(first, attribute), getattr(second, attribute))
    assert first.alpha == second.alpha


def test_canonical_delay_alignment_and_nested_stream_prefixes():
    stream = canonical._inputs(13, 8 + 64, 33)
    delay = canonical._delay_blocks({"train": stream}, window=2, burnin=8)["train"]
    np.testing.assert_array_equal(delay[0], stream[7:9][::-1])
    for count in (8, 16, 32, 64):
        short = canonical._inputs(13, 8 + count, 33)
        np.testing.assert_array_equal(stream[:8 + count], short)
        short_delay = canonical._delay_blocks({"train": short}, window=2, burnin=8)["train"]
        np.testing.assert_array_equal(delay[:count], short_delay)


@pytest.fixture(scope="module")
def smoke_results():
    cfg = {"study": "small_engineering_fixture", "training_rows": [480, 960, 1920, 3840],
           "validation_rows": 512, "test_rows": 2048, "reference_rows": 1024,
           "panel_groups": 32, "prefix_branches": 4, "prefix_length": 128,
           "reference_prefix_lengths": [128, 256], "input_levels": 33, "max_snapshots": 4,
           "alpha_grid": [0., 1e-4, 1., "+inf"], "development_seed": 840091,
           "anchors": [{"id": "never_simulated", "teacher_n_qubits": 4,
                        "student_n_qubits": 6, "window": 6, "gamma": 4.}],
           "roots": [{"root_id": "never_opened", "seed": 840092}]}
    effective = runner.profile_config(cfg, "smoke")
    rows, manifest = runner.run_anchor(effective, "smoke", effective["roots"][0],
                                        effective["anchors"][0], runner.seed_manifest(cfg))
    return cfg, effective, pd.DataFrame(rows), manifest


def test_smoke_all_budgets_fixed_reference_and_exact_panel_identity(smoke_results):
    _, cfg, table, manifest = smoke_results
    audit = runner.validate_coverage(table, cfg, cfg["roots"])
    assert audit == {"passed": True, "expected_rows": 16, "actual_rows": 16}
    assert set(table.loc[table.model == "ptd_parallel", "feature_budget"]) == {9, 18, 36}
    assert set(table.loc[table.model == "poly2_delay", "feature_budget"]) == {5}
    assert table.target_variance.nunique() == table.reference_mean_sha256.nunique() == 1
    assert (table.fit_rows == table.training_rows + 24).all()
    assert (table.train_standardization_rows == table.training_rows).all()
    for name in ("train", "test", "reference_mean", "panel_empirical_mean", "panel_exact_mean"):
        np.testing.assert_allclose(table[f"{name}_nrmse"] ** 2 * table.target_variance,
                                   table[f"{name}_mse"], atol=1e-14)
    np.testing.assert_allclose(table.panel_total_mse,
                              table.panel_history_mse + table.panel_empirical_mean_mse, atol=1e-14)
    np.testing.assert_allclose(table.panel_empirical_minus_exact_mean_mse,
                              table.panel_empirical_mean_mse - table.panel_exact_mean_mse)
    assert table.identity_error.max() < 1e-14
    assert manifest["reference_prefix_stability"]["prefix_lengths"] == [8, 16]
    assert "not stationarity" in manifest["reference_prefix_stability"]["claim_boundary"]
    assert len(manifest["train_slice_input_sha256"]) == 4
    assert len(manifest["array_sha256"]["reference_conditional_means"]) == 64
    assert set(manifest["mean_prefix_state_integrity"]) == {"8", "16"}
    assert all(item["passed"] for item in manifest["mean_prefix_state_integrity"].values())


@pytest.mark.parametrize("change", ["missing", "duplicate", "denominator", "reference", "nonfinite", "fit_rows"])
def test_result_audit_rejects_incomplete_or_mismatched_comparisons(smoke_results, change):
    _, cfg, original, _ = smoke_results
    table = original.copy()
    if change == "missing":
        table = table.iloc[:-1]
    elif change == "duplicate":
        table = pd.concat([table, table.iloc[:1]], ignore_index=True)
    elif change == "denominator":
        table.loc[0, "target_variance"] *= 2
    elif change == "reference":
        table.loc[0, "reference_mean_sha256"] = "different"
    elif change == "nonfinite":
        table.loc[0, "test_nrmse"] = np.nan
    else:
        table.loc[0, "fit_rows"] = 2
    with pytest.raises(RuntimeError):
        runner.validate_coverage(table, cfg, cfg["roots"])


def test_freeze_is_required_and_binds_material_config_and_roots(config, tmp_path, monkeypatch):
    path = write_config(tmp_path, config)
    monkeypatch.setattr(runner, "material_manifest", lambda path=None: {"files": {}, "identity_sha256": "a" * 64})
    with pytest.raises(ValueError, match="requires --freeze"):
        runner.run(path, tmp_path / "no_run", profile="production")
    assert not (tmp_path / "no_run").exists()
    expected = {"status": "FROZEN_FOLLOWUP", "study": config["study"],
                "config_sha256": runner.config_file_sha256(path), "material_sha256": "a" * 64,
                "root_ids": [root["root_id"] for root in config["roots"]],
                "production_roots_previously_opened": False}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(expected))
    assert runner.verify_freeze(path, freeze) == expected
    for key, value in (("material_sha256", "wrong"), ("config_sha256", "wrong"),
                       ("root_ids", []), ("production_roots_previously_opened", 0)):
        changed = {**expected, key: value}
        freeze.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=key):
            runner.verify_freeze(path, freeze)


def test_run_never_overwrites_or_selects_production_root_in_smoke(config, tmp_path, monkeypatch):
    path = write_config(tmp_path, config)
    monkeypatch.setattr(runner, "material_manifest", lambda path=None: {"files": {}, "identity_sha256": "a" * 64})
    output = tmp_path / "existing"
    output.mkdir()
    protected = output / "keep.txt"
    protected.write_text("keep")
    with pytest.raises(ValueError, match="never overwrite"):
        runner.run(path, output)
    assert protected.read_text() == "keep"
    with pytest.raises(ValueError, match="not a root"):
        runner.run(path, tmp_path / "new", root_id="unopened01")


def test_run_serializes_verifiable_root_artifacts(smoke_results, tmp_path, monkeypatch):
    raw, effective, table, anchor_manifest = smoke_results
    path = write_config(tmp_path, raw)
    monkeypatch.setattr(runner, "material_manifest", lambda path=None: {"files": {}, "identity_sha256": "a" * 64})
    monkeypatch.setattr(runner, "run_anchor", lambda *args: (table.to_dict("records"), anchor_manifest))
    output = tmp_path / "result"
    summary = runner.run(path, output)
    assert summary["scientific_evidence"] is False
    assert summary["selected_root_ids"] == ["development_smoke"]
    assert summary["material_sha256"] == "a" * 64
    assert summary["metrics_sha256"] == runner.config_file_sha256(output / "metrics.csv")
    root_manifest = output / "development_smoke" / "manifest.json"
    assert summary["roots"][0]["manifest_sha256"] == runner.config_file_sha256(root_manifest)
    assert json.loads(root_manifest.read_text())["anchors"][0]["array_sha256"] == anchor_manifest["array_sha256"]
    assert not (output / "failure.json").exists()


def test_array_hash_binds_shape_dtype_and_values():
    values = np.arange(6, dtype=np.float64)
    assert runner.array_sha256(values) != runner.array_sha256(values.reshape(2, 3))
    assert runner.array_sha256(values) != runner.array_sha256(values.astype(np.float32))
    assert runner.array_sha256(values) != runner.array_sha256(values + 1)


def test_mean_state_integrity_is_recorded_without_changing_state():
    state = np.array([[.5, -.5j], [.5j, .5]], dtype=complex)
    before = state.copy()
    audit = runner.mean_state_integrity(state)
    assert audit["passed"]
    assert audit["trace_error"] == audit["hermiticity_max_abs_error"] == 0.0
    assert audit["minimum_eigenvalue"] == 0.0
    np.testing.assert_array_equal(state, before)
    # The declared numerical tolerance permits a tiny negative eigenvalue.
    assert runner.mean_state_integrity(np.diag([-5e-9, 1 + 5e-9]))["passed"]


@pytest.mark.parametrize("state", [np.eye(2), np.diag([-2e-8, 1 + 2e-8]),
                                  np.array([[.5, 1e-8j], [0., .5]]),
                                  np.array([[np.nan, 0.], [0., 1.]]),
                                  np.zeros((2, 3))])
def test_mean_state_integrity_rejects_invalid_reference(state):
    with pytest.raises(RuntimeError, match="mean prefix state"):
        runner.mean_state_integrity(state)


@pytest.mark.parametrize("magnitude", [0., 1e-5, 1e-4])
def test_fixed_variance_fails_at_or_below_threshold(magnitude):
    with pytest.raises(RuntimeError, match="strictly greater"):
        runner.fixed_test_target_variance(np.array([[-magnitude], [magnitude]]))


def test_fixed_variance_retains_the_declared_test_only_population_denominator():
    values = np.array([[0., 1.], [2., 7.], [4., 10.]])
    assert runner.fixed_test_target_variance(values) == float(np.mean(np.var(values, axis=0)))
