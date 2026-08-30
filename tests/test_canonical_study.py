import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from qrc_vs_qelm.canonical_gates import verify_duplicate_smoke_runs
from qrc_vs_qelm.canonical_study import (
    _condition_number,
    _inputs,
    _suffix_inputs,
    _suffix_prediction_audit,
    load_canonical_profile,
    run_canonical_study,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "canonical_emulation.yaml"


def test_smoke_profile_is_canonical_and_seed_disjoint():
    profile = load_canonical_profile(CONFIG, "smoke")

    assert profile["confirmatory"] is False
    assert profile["teacher_n_qubits"] == [2]
    assert profile["teacher_dissipators"] == ["local_loss"]
    assert profile["windows"] == [1, 2]
    assert profile["student"]["architecture"] == "ptd_parallel"
    assert profile["student"]["encoding"] == "parallel"
    assert profile["student"]["n_qubits"] == 6
    assert profile["freeze_record_path"].endswith(
        "configs/canonical_emulation_freeze.json"
    )
    assert profile["decision"] == {
        "tolerance_nrmse": 0.10,
        "min_target_variance": 1.0e-8,
        "min_root_successes": 10,
        "roots_total": 12,
        "min_spearman": 0.80,
        "min_g2_live_systems_per_family": 5,
        "min_window_gain": 0.20,
        "min_feature_gain": 0.20,
        "min_best_classical_gain": 0.10,
        "min_scaling_r_squared": 0.85,
        "g4_gammas": [2.0, 4.0],
        "g5_gamma": 4.0,
    }


def test_confirmatory_profile_has_exact_frozen_root_denominator():
    profile = load_canonical_profile(CONFIG, "confirmatory")

    assert profile["status"] == "FROZEN_CONFIRMATORY"
    assert profile["confirmatory"] is True
    assert len(profile["roots"]) == profile["decision"]["roots_total"] == 12
    assert [root["root_id"] for root in profile["roots"]] == [
        f"confirm{index:02d}" for index in range(1, 13)
    ]


def test_confirmatory_run_requires_matching_freeze_record(tmp_path: Path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["status"] = "FROZEN_CONFIRMATORY"
    raw["freeze_record"] = "missing-freeze.json"
    path = tmp_path / "unapproved.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(RuntimeError, match="approved freeze record"):
        run_canonical_study(
            config_path=path,
            profile_name="confirmatory",
            output_dir=tmp_path / "must-not-open",
        )


def test_profile_rejects_global_same_role_seed_reuse(tmp_path: Path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["profiles"]["calibration"]["roots"][1]["student_coupling_seed"] = raw[
        "profiles"
    ]["calibration"]["roots"][0]["student_coupling_seed"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="seed collision"):
        load_canonical_profile(path, "calibration")


def test_profile_rejects_confirmatory_seed_reused_from_smoke(tmp_path: Path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["profiles"]["confirmatory"]["roots"][0]["student_coupling_seed"] = raw[
        "profiles"
    ]["smoke"]["roots"][0]["student_coupling_seed"]
    path = tmp_path / "cross_phase_collision.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="global seed collision"):
        load_canonical_profile(path, "smoke")


def test_profile_rejects_window_wider_than_parallel_register(tmp_path: Path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["profiles"]["smoke"]["windows"] = [1, 7]
    path = tmp_path / "bad_window.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="no larger"):
        load_canonical_profile(path, "smoke")


def test_profile_rejects_invalid_frozen_gate_threshold(tmp_path: Path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["common"]["decision"]["min_scaling_r_squared"] = 1.01
    path = tmp_path / "bad_gate.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="min_scaling_r_squared"):
        load_canonical_profile(path, "smoke")


def test_discrete_inputs_are_deterministic_and_on_the_declared_alphabet():
    first = _inputs(1234, 40, 33)
    second = _inputs(1234, 40, 33)

    assert np.array_equal(first, second)
    assert np.all(first >= 0.0)
    assert np.all(first <= 1.0)
    assert np.allclose(first * 32, np.round(first * 32))


def test_resolved_rank_one_pseudocondition_is_one():
    feature = np.arange(12.0)[:, None]

    assert _condition_number(feature) == pytest.approx(1.0)
    assert np.isinf(_condition_number(np.ones((12, 2))))


def test_suffix_panels_are_deterministic_nested_tails():
    profile = load_canonical_profile(CONFIG, "calibration")
    root = profile["roots"][0]

    prefixes_first, suffixes_first = _suffix_inputs(profile, root)
    prefixes_second, suffixes_second = _suffix_inputs(profile, root)
    longest = suffixes_first[max(profile["windows"])]

    assert np.array_equal(prefixes_first, prefixes_second)
    for window in profile["windows"]:
        assert np.array_equal(suffixes_first[window], suffixes_second[window])
        assert np.array_equal(suffixes_first[window], longest[:, -window:])
        assert len(np.unique(suffixes_first[window], axis=0)) == profile[
            "suffix_groups"
        ]


def test_profile_rejects_more_suffix_groups_than_alphabet_symbols(tmp_path: Path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["profiles"]["smoke"]["suffix_groups"] = 34
    path = tmp_path / "duplicate_suffix_risk.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="suffix_groups cannot exceed"):
        load_canonical_profile(path, "smoke")


def test_suffix_prediction_audit_checks_both_common_suffix_bounds():
    targets = np.array([[[-1.0], [1.0]]])
    prediction = np.array([[0.0]])
    repeated = np.array([[[0.0], [0.0]]])
    pair_diagnostics = {
        "max_group_diameter": 2.0,
        "rms_pair_half_bound": 1.0,
    }

    result = _suffix_prediction_audit(
        targets,
        prediction,
        repeated,
        pair_diagnostics,
    )

    assert result["prediction_disagreement"] == 0.0
    assert result["repeated_suffix_rmse"] == pytest.approx(1.0)
    assert result["worst_branch_rmse"] == pytest.approx(1.0)
    assert result["rms_bound_slack"] == pytest.approx(0.0)
    assert result["diameter_bound_slack"] == pytest.approx(0.0)
    assert result["predictions_agree"]
    assert result["rms_half_bound_satisfied"]
    assert result["diameter_half_bound_satisfied"]
    assert result["suffix_bounds_satisfied"]


def test_suffix_prediction_audit_rejects_prefix_dependent_predictions():
    targets = np.array([[[-1.0], [1.0]]])
    repeated = targets.copy()

    result = _suffix_prediction_audit(
        targets,
        np.array([[-1.0]]),
        repeated,
        {
            "max_group_diameter": 2.0,
            "rms_pair_half_bound": 1.0,
        },
    )

    assert result["prediction_disagreement"] == pytest.approx(2.0)
    assert not result["predictions_agree"]
    assert not result["rms_half_bound_satisfied"]
    assert not result["diameter_half_bound_satisfied"]
    assert not result["suffix_bounds_satisfied"]


def test_suffix_prediction_audit_compares_vectorized_base_to_independent_arms():
    targets = np.array([[[-1.0], [1.0]]])
    repeated = np.zeros_like(targets)

    result = _suffix_prediction_audit(
        targets,
        np.array([[1.0]]),
        repeated,
        {
            "max_group_diameter": 2.0,
            "rms_pair_half_bound": 1.0,
        },
    )

    assert result["prediction_disagreement"] == pytest.approx(1.0)
    assert not result["predictions_agree"]


def test_smoke_runner_records_production_suffix_audits(tmp_path: Path):
    output = tmp_path / "canonical-smoke"

    summary = run_canonical_study(
        config_path=CONFIG,
        profile_name="smoke",
        output_dir=output,
    )
    suffix = pd.read_csv(output / "suffix_decomposition.csv")
    stream = pd.read_csv(output / "stream_metrics.csv")
    systems = pd.read_csv(output / "system_diagnostics.csv")
    resources = json.loads(
        (output / "computational_resources.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output / "canonical_manifest.json").read_text(encoding="utf-8")
    )

    assert summary["max_decomposition_identity_error"] <= 1e-12
    assert suffix["predictions_agree"].astype(bool).all()
    assert suffix["rms_half_bound_satisfied"].astype(bool).all()
    assert suffix["diameter_half_bound_satisfied"].astype(bool).all()
    assert suffix["suffix_bounds_satisfied"].astype(bool).all()
    assert (suffix["prediction_disagreement"] <= 1e-12).all()
    assert "intercept_only_selected" in stream.columns
    assert np.allclose(
        2.0 * suffix["half_max_group_diameter"],
        suffix["max_group_diameter"],
        atol=1e-12,
    )
    assert {
        "target_observables_sha256",
        "student_n_qubits",
        "student_observables_sha256",
    } <= set(stream.columns)
    assert {"target_dim", "target_observables_sha256"} <= set(systems.columns)
    assert resources["decision_bearing"] is False
    assert resources["runtime"]["wall_clock_seconds"] > 0.0
    assert resources["measurement_setting_proxies"][
        "student_global_basis_settings_per_budget"
    ] == {"1": 3, "2": 6}
    assert "computational_resources.json" in manifest["artifacts"]
    material_identity = manifest["reproducibility"]["material"]["identity_sha256"]
    same_directory = verify_duplicate_smoke_runs(
        output,
        output,
        expected_material_identity_sha256=material_identity,
    )
    assert not same_directory["passed"]
    assert not same_directory["distinct_directories"]
