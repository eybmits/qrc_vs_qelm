from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = ROOT / "scripts" / "run_emulation_pilot.py"
    spec = importlib.util.spec_from_file_location("run_emulation_pilot_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def test_frozen_smoke_profile_is_valid_and_seed_disjoint():
    study = RUNNER.load_profile(
        ROOT / "configs" / "emulation_pilot.yaml", "smoke"
    )

    assert study["status"] == "EXPLORATORY_PILOT"
    assert study["profile"]["windows"] == (1, 3, 8)
    teacher = {
        int(row["teacher_coupling_seed"])
        for row in study["profile"]["systems"]
    }
    assert teacher.isdisjoint(study["profile"]["student_coupling_seeds"])


def test_profile_rejects_teacher_student_seed_overlap(tmp_path: Path):
    source = (ROOT / "configs" / "emulation_pilot.yaml").read_text(
        encoding="utf-8"
    )
    bad = source.replace("teacher_coupling_seed: 910001", "teacher_coupling_seed: 915001", 1)
    path = tmp_path / "bad.yaml"
    path.write_text(bad, encoding="utf-8")

    with pytest.raises(ValueError, match="must be disjoint"):
        RUNNER.load_profile(path, "smoke")


def test_matched_profile_holds_all_nuisance_seeds_fixed_within_block():
    study = RUNNER.load_profile(
        ROOT / "configs" / "emulation_pilot.yaml", "matched_pilot"
    )

    assert study["profile"]["matched_gamma_design"]
    systems = pd.DataFrame(study["profile"]["systems"])
    for _, group in systems.groupby("replicate_id"):
        assert group["gamma"].tolist() == [0.0, 0.05, 0.2, 0.5]
        for column in (
            "teacher_coupling_seed",
            "train_input_seed",
            "val_input_seed",
            "test_input_seed",
            "pair_seed",
        ):
            assert group[column].nunique() == 1


def test_replicated_profile_uses_independent_nested_bank_per_block():
    study = RUNNER.load_profile(
        ROOT / "configs" / "emulation_pilot.yaml", "replicated_pilot"
    )

    assert study["profile"]["independent_student_banks"]
    banks = {}
    for system in study["profile"]["systems"]:
        replicate_id = system["replicate_id"]
        bank = tuple(system["student_coupling_seeds"])
        assert len(bank) == 4
        if replicate_id in banks:
            assert banks[replicate_id] == bank
        banks[replicate_id] = bank
    assert len(banks) == 4
    assert len(set(banks.values())) == 4
    assert set().union(*(set(bank) for bank in banks.values())).isdisjoint(
        {int(system["teacher_coupling_seed"]) for system in study["profile"]["systems"]}
    )
    wide = RUNNER.load_profile(
        ROOT / "configs" / "emulation_pilot.yaml",
        "replicated_pilot_wide_ridge",
    )
    assert max(wide["profile"]["alpha_grid"]) == 1.0e6
    complete = RUNNER.load_profile(
        ROOT / "configs" / "emulation_pilot.yaml",
        "replicated_pilot_intercept_complete",
    )
    assert np.isposinf(max(complete["profile"]["alpha_grid"]))


def test_matched_profile_rejects_physical_nuisance_change(tmp_path: Path):
    source = ROOT / "configs" / "emulation_pilot.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["profiles"]["matched_pilot"]["systems"][1]["h"] = 0.75
    path = tmp_path / "physical-confound.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="physical nuisances"):
        RUNNER.load_profile(path, "matched_pilot")


def test_profile_rejects_seed_namespace_collision(tmp_path: Path):
    source = ROOT / "configs" / "emulation_pilot.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    system = payload["profiles"]["smoke"]["systems"][0]
    system["train_input_seed"] = system["teacher_coupling_seed"]
    path = tmp_path / "seed-collision.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="seed namespaces"):
        RUNNER.load_profile(path, "smoke")


def test_production_window_path_ignores_prefix_without_copying_predictions():
    study = RUNNER.load_profile(
        ROOT / "configs" / "emulation_pilot.yaml", "smoke"
    )
    system = study["profile"]["systems"][0]
    cfg = RUNNER._qrc_config(study["common"], system)
    observables = RUNNER._observables(study["common"], cfg.n_qubits)
    reservoir = RUNNER._reservoir(cfg, int(system["teacher_coupling_seed"]))
    first = RUNNER.quantize_inputs(
        np.array([0.0, 0.2, 0.4, 0.1, 0.5, 0.8]), cfg.quantize_levels
    )
    second = RUNNER.quantize_inputs(
        np.array([1.0, 0.9, 0.7, 0.1, 0.5, 0.8]), cfg.quantize_levels
    )

    feature_a = RUNNER._production_replay_feature(
        reservoir, first, 3, observables
    )
    feature_b = RUNNER._production_replay_feature(
        reservoir, second, 3, observables
    )

    assert np.array_equal(feature_a, feature_b)
    qelm_cfg = RUNNER.windowed_qelm_from_qrc(cfg, window_k=3)
    production_bundle = RUNNER.build_features(
        qelm_cfg,
        first,
        int(system["teacher_coupling_seed"]),
        observables=observables,
    )
    assert np.allclose(feature_a, production_bundle.features[-1], atol=1e-12)


def test_classical_inputs_are_on_the_same_quantized_alphabet():
    study = RUNNER.load_profile(
        ROOT / "configs" / "emulation_pilot.yaml", "smoke"
    )
    system = study["profile"]["systems"][0]
    blocks = RUNNER._input_blocks(
        study["profile"],
        system,
        valid_prefix=8,
        quantize_levels=33,
    )

    for values in blocks.values():
        assert np.allclose(values * 32, np.rint(values * 32), atol=1e-12)


def test_evaluate_decision_detects_clean_synthetic_go():
    stream_rows = []
    certificate_rows = []
    systems = pd.DataFrame(
        [
            {"system_id": "u", "replicate_id": "r", "gamma": 0.0, "D2_d0": 1.0},
            {"system_id": "a", "replicate_id": "r", "gamma": 0.2, "D2_d0": 1.0},
            {"system_id": "b", "replicate_id": "r", "gamma": 0.5, "D2_d0": 1.0},
        ]
    )
    for system_id, gamma in (("u", 0.0), ("a", 0.2), ("b", 0.5)):
        curves = {
            "oracle_identity": [1.0, 0.5 if gamma else 0.95, 0.1 if gamma else 0.9],
            "random_bank_R1": [1.0, 0.7, 0.5 if gamma else 0.9],
            "random_bank_R4": [1.0, 0.3, 0.1 if gamma else 0.8],
            "linear_delay": [1.0, 0.5, 0.3 if gamma else 0.85],
        }
        for model, values in curves.items():
            for window, nrmse in zip((1, 4, 8), values):
                stream_rows.append(
                    {"system_id": system_id, "gamma": gamma, "model": model, "window": window, "nrmse": nrmse}
                )
        separations = [1.0, 0.5 if gamma else 0.95, 0.1 if gamma == 0.5 else (0.2 if gamma else 0.9)]
        for window, separation in zip((1, 4, 8), separations):
            certificate_rows.append(
                {
                    "system_id": system_id,
                    "gamma": gamma,
                    "model": "oracle_identity",
                    "window": window,
                    "rms_suffix_separation": separation,
                    "paired_rmse_at_witness": separation,
                    "D2_fraction_of_d0": separation,
                    "bound_satisfied": True,
                    "prediction_disagreement": 0.0,
                }
            )
    trace = pd.DataFrame(
        [
            {"system_id": system_id, "suffix_endpoint": endpoint, "trace_distance_mean": value}
            for system_id, final in (("u", 1.0), ("a", 0.1), ("b", 0.01))
            for endpoint, value in ((0, 1.0), (8, final))
        ]
    )
    decision = {
        "numeric_tolerance": 1e-10,
        "liveness_D2_at_d0": 1e-3,
        "min_live_system_fraction": 0.66,
        "min_oracle_error_certificate_spearman": 0.8,
        "min_associated_dissipative_system_fraction": 0.5,
        "min_oracle_max_window_reduction": 0.5,
        "min_dissipative_system_fraction_improved": 0.5,
        "generic_student_budget": 4,
        "generic_student_max_window_nrmse": 0.25,
        "generic_student_min_reduction": 0.5,
        "min_feature_budget_reduction": 0.15,
        "min_feature_budget_system_fraction": 0.5,
        "min_generic_gain_over_linear": 0.1,
        "min_generic_over_linear_system_fraction": 0.5,
        "nonfading_min_remaining_trace_fraction": 0.9,
        "min_nonfading_system_fraction": 0.5,
        "max_matched_gamma_memory_spearman": -0.8,
        "min_matched_replicate_fraction_ordered": 0.5,
    }

    result = RUNNER.evaluate_decision(
        pd.DataFrame(stream_rows),
        pd.DataFrame(certificate_rows),
        systems,
        trace,
        decision,
        matched_gamma_design=True,
    )

    assert result["E0_suffix_certificate_implementation"]
    assert result["E3_constructive_convergence"]
    assert result["E4a_generic_window_emulation"]
    assert result["E4b_feature_budget_gain"]
    assert result["E4c_generic_beats_linear"]
    assert result["E5_nonfading_boundary"]
    assert result["E6_matched_mixing_order"]
    assert result["verdict"] == "PILOT_SUPPORTS_TWO_RESOURCE_SCALING"
    json.dumps(result, allow_nan=False)

    unmatched = RUNNER.evaluate_decision(
        pd.DataFrame(stream_rows),
        pd.DataFrame(certificate_rows),
        systems,
        trace,
        decision,
        matched_gamma_design=False,
    )
    assert unmatched["E6_matched_mixing_order"] is None
