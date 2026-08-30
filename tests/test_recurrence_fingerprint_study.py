from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER = ROOT / "src" / "qrc_vs_qelm" / "recurrence_fingerprint.py"
SCRIPT = ROOT / "scripts" / "run_recurrence_fingerprint_study.py"

if not CLASSIFIER.exists():
    pytest.skip(
        "qrc_vs_qelm.recurrence_fingerprint has not landed yet",
        allow_module_level=True,
    )

spec = importlib.util.spec_from_file_location(
    "run_recurrence_fingerprint_study",
    SCRIPT,
)
study = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = study
spec.loader.exec_module(study)


def _minimal_config() -> dict:
    return {
        "schema_version": 1,
        "freeze_name": "test_fixture",
        "scope": {"max_window": 5},
        "probe": {},
        "measurement": {},
        "classifier": {},
        "task_link": {},
        "design_blocks": [
            {"physics_id": "design_a", "coupling_seed": 101},
            {"physics_id": "design_b", "coupling_seed": 102},
        ],
        "heldout_blocks": [
            {"physics_id": "heldout_a", "coupling_seed": 201},
            {"physics_id": "heldout_b", "coupling_seed": 202},
        ],
    }


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_locked_config_validates_required_keys_and_disjointness(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid.json"
    valid = _minimal_config()
    _write_config(valid_path, valid)

    loaded, lock_hash = study.load_locked_config(valid_path)

    assert loaded == valid
    assert lock_hash == study.manifests.sha256_file(valid_path)

    missing = copy.deepcopy(valid)
    missing.pop("probe")
    missing_path = tmp_path / "missing.json"
    _write_config(missing_path, missing)
    with pytest.raises(ValueError, match="missing keys.*probe"):
        study.load_locked_config(missing_path)

    physics_overlap = copy.deepcopy(valid)
    physics_overlap["heldout_blocks"][0]["physics_id"] = "design_a"
    physics_path = tmp_path / "physics_overlap.json"
    _write_config(physics_path, physics_overlap)
    with pytest.raises(ValueError, match="design/heldout physics overlap"):
        study.load_locked_config(physics_path)

    seed_overlap = copy.deepcopy(valid)
    seed_overlap["heldout_blocks"][0]["coupling_seed"] = 101
    seed_path = tmp_path / "seed_overlap.json"
    _write_config(seed_path, seed_overlap)
    with pytest.raises(ValueError, match="coupling seeds must be disjoint"):
        study.load_locked_config(seed_path)

    wrong_window = copy.deepcopy(valid)
    wrong_window["scope"]["max_window"] = 4
    window_path = tmp_path / "wrong_window.json"
    _write_config(window_path, wrong_window)
    with pytest.raises(ValueError, match="locked to max_window=5"):
        study.load_locked_config(window_path)


def test_v1_config_hash_and_declared_observable_dimensions_match_execution(
    tmp_path: Path,
) -> None:
    config_path = ROOT / "data" / "frozen" / "recurrence_fingerprint_v1.json"
    config, lock_hash = study.load_locked_config(config_path)

    assert lock_hash == study.V1_CONFIG_SHA256
    declared = config["scope"]["observables"]["dimensions_by_n"]
    for n_qubits in (2, 3, 4):
        observables = study.readout.pauli_observables(
            n_qubits,
            axes=("x", "y", "z"),
            max_weight=2,
        )
        assert len(observables) == int(declared[str(n_qubits)])

    modified = copy.deepcopy(config)
    modified["scope"]["observables"]["family"] = "altered"
    modified_path = tmp_path / "modified_v1.json"
    _write_config(modified_path, modified)
    with pytest.raises(ValueError, match="frozen SHA-256"):
        study.load_locked_config(modified_path)


def test_blind_id_is_deterministic_and_source_separating() -> None:
    lock_hash = "a" * 64
    expected = study._blind_id(lock_hash, "physics_1", "qrc")

    assert study._blind_id(lock_hash, "physics_1", "qrc") == expected
    assert expected.startswith("case_")
    assert len(expected) == len("case_") + 16
    assert len(
        {
            expected,
            study._blind_id("b" * 64, "physics_1", "qrc"),
            study._blind_id(lock_hash, "physics_2", "qrc"),
            study._blind_id(lock_hash, "physics_1", "replay_qelm_d1"),
        }
    ) == 4


def test_collect_block_keeps_source_metadata_out_of_blinded_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix_lengths = (0, 1, 3)
    profile = SimpleNamespace(
        qelm_specs=(SimpleNamespace(model_name="replay_qelm_d1", window=1),),
        pairs_per_coupling=1,
        suffix_lengths=suffix_lengths,
    )
    monkeypatch.setattr(study, "_profile_for_block", lambda config, block: profile)
    monkeypatch.setattr(
        study.POC,
        "_build_dynamics",
        lambda profile, coupling_seed: (object(), object(), ("Z0", "Z1")),
    )

    def fake_exact_curves(*args, **kwargs):
        curves = {}
        for length in suffix_lengths:
            carry_a = np.array([0.5 + 0.1 * length, -0.2])
            carry_b = np.array([0.1, -0.1])
            reset_a = np.array([0.0, 0.0])
            reset_b = np.array([0.0, 0.0])
            curves[("qrc", "carry", length)] = (carry_a, carry_b)
            curves[("qrc", "ideal_quantum_reset", length)] = (
                reset_a,
                reset_b,
            )
            curves[("replay_qelm_d1", "carry", length)] = (
                carry_a / 2.0,
                carry_b / 2.0,
            )
            curves[("replay_qelm_d1", "ideal_quantum_reset", length)] = (
                carry_a / 2.0,
                carry_b / 2.0,
            )
        return curves

    monkeypatch.setattr(study.POC, "_exact_curves_for_pair", fake_exact_curves)

    def fake_cross_batch(x_a, x_b, *, tier, seed, parts):
        value = float(np.mean((x_a - x_b) ** 2))
        return value, value

    monkeypatch.setattr(study.POC, "_cross_batch_estimate", fake_cross_batch)
    config = {
        "scope": {
            "max_window": 5,
            "observables": {
                "dimensions_by_n": {"2": 9},
            },
        },
        "measurement": {
            "shots_per_batch": [64],
            "measurement_seeds": [7],
        },
    }
    block = {"physics_id": "physics_a", "coupling_seed": 11}

    scores, key = study._collect_block(config, "c" * 64, block)

    assert study.FORBIDDEN_CLASSIFIER_COLUMNS.isdisjoint(scores.columns)
    assert study.FORBIDDEN_CLASSIFIER_COLUMNS <= set(key.columns)
    assert key["blind_id"].is_unique
    assert set(scores["blind_id"]) == set(key["blind_id"])
    assert set(key["source_model"]) == {
        "qrc",
        "replay_qelm_d1",
        "forced_reset_qrc_d3",
        "leaky_qelm_control",
        "reset_failure_qrc_control",
        "dead_readout_control",
    }


def test_fault_curve_augmentation_has_the_declared_behaviors() -> None:
    suffix_lengths = (0, 2, 3, 5)
    base = {}
    for length in suffix_lengths:
        base[("qrc", "carry", length)] = (
            np.array([1.0 + length, -0.5]),
            np.array([-1.0, 0.25]),
        )
        base[("qrc", "ideal_quantum_reset", length)] = (
            np.array([0.2, 0.2]),
            np.array([0.2, 0.2]),
        )

    augmented = study._augmented_curve_pairs(base, suffix_lengths)

    for length in suffix_lengths:
        qrc_carry = base[("qrc", "carry", length)]
        qrc_reset = base[("qrc", "ideal_quantum_reset", length)]
        forced = augmented[("forced_reset_qrc_d3", "carry", length)]
        forced_source = qrc_carry if length < 3 else qrc_reset
        assert np.array_equal(forced[0], forced_source[0])
        assert np.array_equal(forced[1], forced_source[1])

        forced_reset = augmented[
            ("forced_reset_qrc_d3", "ideal_quantum_reset", length)
        ]
        assert np.array_equal(forced_reset[0], qrc_reset[0])
        assert np.array_equal(forced_reset[1], qrc_reset[1])

        leaky_carry = augmented[("leaky_qelm_control", "carry", length)]
        leaky_reset = augmented[
            ("leaky_qelm_control", "ideal_quantum_reset", length)
        ]
        assert np.array_equal(leaky_carry[0], qrc_carry[0])
        assert np.array_equal(leaky_reset[0], qrc_reset[0])

        failed_reset = augmented[
            ("reset_failure_qrc_control", "ideal_quantum_reset", length)
        ]
        assert np.array_equal(failed_reset[0], qrc_carry[0])
        assert np.array_equal(failed_reset[1], qrc_carry[1])

        for condition in ("carry", "ideal_quantum_reset"):
            dead = augmented[("dead_readout_control", condition, length)]
            assert np.array_equal(dead[0], np.zeros_like(qrc_carry[0]))
            assert np.array_equal(dead[1], np.zeros_like(qrc_carry[1]))

    copied = augmented[("forced_reset_qrc_d3", "carry", 0)][0]
    assert not np.shares_memory(copied, base[("qrc", "carry", 0)][0])


def test_prediction_family_and_primary_metrics_are_mapped_exactly() -> None:
    assert study._prediction_family("FINITE_WINDOW_COMPATIBLE_d3") == "finite_window"
    assert study._prediction_family("RECURRENT_RESET_SENSITIVE") == (
        "recurrent_reset_sensitive"
    )
    assert study._prediction_family("RECURRENT_RESET_INSENSITIVE") == (
        "recurrent_reset_insensitive"
    )
    assert study._prediction_family("INDETERMINATE_LOW_SIGNAL") == "low_signal"
    assert study._prediction_family("UNRECOGNIZED") == "indeterminate"

    frame = pd.DataFrame(
        [
            {
                "primary_case": True,
                "expected_behavior": "recurrent_reset_sensitive",
                "predicted_family": "recurrent_reset_sensitive",
                "actual_window": np.nan,
                "inferred_cutoff": np.nan,
            },
            {
                "primary_case": True,
                "expected_behavior": "recurrent_reset_sensitive",
                "predicted_family": "low_signal",
                "actual_window": np.nan,
                "inferred_cutoff": np.nan,
            },
            {
                "primary_case": True,
                "expected_behavior": "finite_window",
                "predicted_family": "finite_window",
                "actual_window": 1,
                "inferred_cutoff": 2,
            },
            {
                "primary_case": True,
                "expected_behavior": "finite_window",
                "predicted_family": "recurrent_reset_sensitive",
                "actual_window": 3,
                "inferred_cutoff": 7,
            },
            {
                "primary_case": True,
                "expected_behavior": "finite_window",
                "predicted_family": "indeterminate",
                "actual_window": 5,
                "inferred_cutoff": 5,
            },
            {
                "primary_case": False,
                "expected_behavior": "finite_window",
                "predicted_family": "finite_window",
                "actual_window": 1,
                "inferred_cutoff": 1,
            },
        ]
    )

    metrics = study._metric_point(frame)

    assert metrics["recurrent_sensitivity"] == pytest.approx(0.5)
    assert metrics["finite_specificity"] == pytest.approx(1.0 / 3.0)
    assert metrics["balanced_accuracy"] == pytest.approx(5.0 / 12.0)
    assert metrics["coverage"] == pytest.approx(0.6)
    assert metrics["correct_rate"] == pytest.approx(0.4)
    assert metrics["error_rate"] == pytest.approx(0.2)
    assert metrics["abstention_rate"] == pytest.approx(0.4)
    assert metrics["false_recurrent_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["cutoff_within_one"] == pytest.approx(2.0 / 3.0)


def test_false_recurrent_counts_both_recurrent_labels_and_reports_counts() -> None:
    frame = pd.DataFrame(
        [
            {
                "primary_case": True,
                "expected_behavior": "recurrent_reset_sensitive",
                "predicted_family": "recurrent_reset_sensitive",
                "actual_window": np.nan,
                "inferred_cutoff": np.nan,
            },
            {
                "primary_case": True,
                "expected_behavior": "finite_window",
                "predicted_family": "finite_window",
                "actual_window": 1,
                "inferred_cutoff": 1,
            },
            {
                "primary_case": True,
                "expected_behavior": "finite_window",
                "predicted_family": "recurrent_reset_insensitive",
                "actual_window": 3,
                "inferred_cutoff": np.nan,
            },
        ]
    )

    point = study._metric_point(frame)
    counts = study._metric_counts(frame)

    assert point["false_recurrent_rate"] == pytest.approx(0.5)
    assert point["coverage"] == pytest.approx(1.0)
    assert counts["false_recurrent_numerator"] == 1
    assert counts["false_recurrent_denominator"] == 2
    assert counts["correct_numerator"] == 2
    assert counts["correct_denominator"] == 3


def test_strict_exact_agreement_counts_abstentions_as_disagreement() -> None:
    frame = pd.DataFrame(
        [
            {
                "classification": "FINITE_WINDOW_COMPATIBLE_d1",
                "exact_classification": "FINITE_WINDOW_COMPATIBLE_d1",
            },
            {
                "classification": "RECURRENT_RESET_SENSITIVE",
                "exact_classification": "FINITE_WINDOW_COMPATIBLE_d1",
            },
            {
                "classification": "INDETERMINATE",
                "exact_classification": "INDETERMINATE",
            },
            {
                "classification": "RECURRENT_RESET_SENSITIVE",
                "exact_classification": "INDETERMINATE_LOW_SIGNAL",
            },
        ]
    )

    point, numerator, denominator = study._strict_exact_agreement_stats(frame)

    assert point == pytest.approx(0.25)
    assert numerator == 1
    assert denominator == 4


def test_heldout_opening_manifest_is_written_once_before_collection(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "locked.json"
    _write_config(config_path, _minimal_config())
    out_dir = tmp_path / "heldout"

    opening = study._write_opening_manifest(
        out_dir=out_dir,
        split="heldout",
        config_path=config_path,
        lock_hash=study.manifests.sha256_file(config_path),
        allow_dirty=True,
    )

    payload = json.loads(opening.read_text(encoding="utf-8"))
    assert payload["core"]["payload"]["opened_before_collection"] is True
    assert payload["core"]["payload"]["split"] == "heldout"
    verified = study._validate_opening_manifest(
        opening_manifest_path=opening,
        config_path=config_path,
        lock_hash=study.manifests.sha256_file(config_path),
    )
    assert verified["pass"] is True
    with pytest.raises(SystemExit, match="open-once"):
        study._write_opening_manifest(
            out_dir=out_dir,
            split="heldout",
            config_path=config_path,
            lock_hash=study.manifests.sha256_file(config_path),
            allow_dirty=True,
        )

    changed = _minimal_config()
    changed["scope"]["max_window"] = 4
    _write_config(config_path, changed)
    with pytest.raises(RuntimeError, match="opening provenance seal failure"):
        study._validate_opening_manifest(
            opening_manifest_path=opening,
            config_path=config_path,
            lock_hash=payload["core"]["payload"]["config_sha256"],
        )


def _integrity_fixture() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = {
        "scope": {
            "observables": {
                "dimensions_by_n": {"2": 9},
            },
        },
        "probe": {
            "histories_per_system": 2,
            "suffix_lengths": [0, 1],
            "conditions": ["carry", "ideal_quantum_reset"],
        },
        "measurement": {
            "shots_per_batch": [10],
            "measurement_seeds": [7],
        },
        "device_panel": {
            "case_count_per_block": 2,
            "cases": [
                {
                    "id": "qrc",
                    "expected_behavior": "recurrent_reset_sensitive",
                    "primary_case": True,
                },
                {
                    "id": "replay_qelm_d1",
                    "effective_window": 1,
                    "expected_behavior": "finite_window",
                    "primary_case": True,
                },
            ],
        },
        "task_link": {"taus": [1, 2]},
        "design_blocks": [{"physics_id": "p1", "n_qubits": 2}],
        "heldout_blocks": [{"physics_id": "h1", "n_qubits": 2}],
    }
    key = pd.DataFrame(
        [
            {
                "blind_id": "bqrc",
                "physics_id": "p1",
                "source_model": "qrc",
                "curve_model": "qrc",
                "expected_behavior": "recurrent_reset_sensitive",
                "primary_case": True,
                "actual_window": np.nan,
            },
            {
                "blind_id": "bqelm",
                "physics_id": "p1",
                "source_model": "replay_qelm_d1",
                "curve_model": "replay_qelm_d1",
                "expected_behavior": "finite_window",
                "primary_case": True,
                "actual_window": 1,
            },
        ]
    )
    score_rows = []
    for blind_id in ("bqrc", "bqelm"):
        for history_id in (0, 1):
            for condition in ("carry", "ideal_quantum_reset"):
                for suffix_length in (0, 1):
                    for shots, measurement_seed in (("infinity", 0), ("10", 7)):
                        score_rows.append(
                            {
                                "blind_id": blind_id,
                                "physics_id": "p1",
                                "history_id": history_id,
                                "condition": condition,
                                "suffix_length": suffix_length,
                                "shots_per_batch": shots,
                                "measurement_seed": measurement_seed,
                                "n_observables": 9,
                                "cross_batch_signal_mse": 0.1,
                            }
                        )
    scores = pd.DataFrame(score_rows)
    prediction_rows = []
    for blind_id in ("bqrc", "bqelm"):
        for shots, measurement_seed in (("infinity", 0), ("10", 7)):
            prediction_rows.append(
                {
                    "blind_id": blind_id,
                    "physics_id": "p1",
                    "shots_per_batch": shots,
                    "measurement_seed": measurement_seed,
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    task_scores = pd.DataFrame(
        [
            {
                "physics_id": "p1",
                "model": model,
                "tau": tau,
                "capacity_test": 0.1,
            }
            for model in ("qrc", "replay_qelm_d5")
            for tau in (1, 2)
        ]
    )
    task_systems = pd.DataFrame([{"physics_id": "p1"}])
    return config, scores, key, predictions, task_scores, task_systems


def test_integrity_requires_exact_cartesian_rows_and_matching_ids() -> None:
    config, scores, key, predictions, task_scores, task_systems = _integrity_fixture()
    result = study.validate_study_integrity(
        scores=scores,
        key=key,
        predictions=predictions,
        task_scores=task_scores,
        task_systems=task_systems,
        config=config,
        split="design",
    )
    assert result["pass"] is True

    duplicated_score = scores.copy()
    duplicated_score.iloc[-1] = duplicated_score.iloc[-2]
    with pytest.raises(RuntimeError, match="score_rows_unique"):
        study.validate_study_integrity(
            scores=duplicated_score,
            key=key,
            predictions=predictions,
            task_scores=task_scores,
            task_systems=task_systems,
            config=config,
            split="design",
        )

    mismatched_prediction = predictions.copy()
    mismatched_prediction.loc[mismatched_prediction.index[-1], "blind_id"] = "extra"
    with pytest.raises(RuntimeError, match="prediction_key_id_pairs"):
        study.validate_study_integrity(
            scores=scores,
            key=key,
            predictions=mismatched_prediction,
            task_scores=task_scores,
            task_systems=task_systems,
            config=config,
            split="design",
        )

    duplicated_task = task_scores.copy()
    duplicated_task.iloc[-1] = duplicated_task.iloc[-2]
    with pytest.raises(RuntimeError, match="task_score_rows_unique"):
        study.validate_study_integrity(
            scores=scores,
            key=key,
            predictions=predictions,
            task_scores=duplicated_task,
            task_systems=task_systems,
            config=config,
            split="design",
        )

    wrong_observable_width = scores.copy()
    wrong_observable_width["n_observables"] = 15
    with pytest.raises(RuntimeError, match="observable_dimensions"):
        study.validate_study_integrity(
            scores=wrong_observable_width,
            key=key,
            predictions=predictions,
            task_scores=task_scores,
            task_systems=task_systems,
            config=config,
            split="design",
        )


def test_heldout_run_rejects_alternate_output_directories(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="canonical output directory"):
        study.run_study(
            config_path=tmp_path / "unused.json",
            split="heldout",
            out_dir=tmp_path / "alternate-heldout",
            allow_dirty=True,
        )


def test_post_opening_exception_writes_final_no_go_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, *_ = _integrity_fixture()
    config.update(
        {
            "schema_version": 1,
            "freeze_name": "failure_fixture",
            "scope": {"max_window": 5},
            "classifier": {},
        }
    )
    config["design_blocks"][0]["coupling_seed"] = 101
    config["heldout_blocks"][0]["coupling_seed"] = 201
    config_path = tmp_path / "failure_config.json"
    _write_config(config_path, config)
    out_dir = tmp_path / "failed_design"
    monkeypatch.setattr(
        study,
        "collect_split",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FloatingPointError("injected numerical failure")
        ),
    )

    paths = study.run_study(
        config_path=config_path,
        split="design",
        out_dir=out_dir,
        allow_dirty=True,
    )

    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    ledger = pd.read_csv(paths["failure_ledger"])
    assert result["execution_status"] == "FAILED_CLOSED"
    assert result["overall_verdict"] == "NO_GO"
    assert result["integrity"]["pass"] is False
    assert result["failure"]["stage"] == "collection"
    assert result["row_accounting"]["scores"]["observed"] == 0
    assert result["row_accounting"]["scores"]["missing"] > 0
    assert set(ledger["kind"]) == {"exception", "row_accounting"}
    assert paths["manifest"].exists()


def test_task_link_summary_uses_long_minus_short_crossover_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        study,
        "QRCConfig",
        lambda **kwargs: SimpleNamespace(kind="qrc", **kwargs),
    )
    monkeypatch.setattr(
        study,
        "windowed_qelm_from_qrc",
        lambda cfg, *, window_k: SimpleNamespace(kind="qelm", window_k=window_k),
    )
    monkeypatch.setattr(
        study.readout,
        "pauli_observables",
        lambda *args, **kwargs: ("Z0",),
    )
    monkeypatch.setattr(study, "assert_matched", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        study.task_families,
        "build_family_targets",
        lambda inputs, taus, degrees: object(),
    )

    def fake_build_features(cfg, inputs, coupling_seed, observables):
        model_code = 0 if cfg.kind == "qrc" else 1
        return SimpleNamespace(
            features=np.array([[float(coupling_seed), float(model_code)]]),
            first_valid=0,
        )

    monkeypatch.setattr(study, "build_features", fake_build_features)
    monkeypatch.setattr(
        study.pipeline,
        "stage_valid_indices",
        lambda *args, **kwargs: np.arange(4),
    )
    monkeypatch.setattr(
        study.pipeline,
        "make_splits",
        lambda *args, **kwargs: object(),
    )

    capacities = {
        (11, 0, 1): 0.4,
        (11, 1, 1): 0.6,
        (11, 0, 6): 0.8,
        (11, 1, 6): 0.5,
        (22, 0, 1): 0.5,
        (22, 1, 1): 0.6,
        (22, 0, 6): 0.7,
        (22, 1, 6): 0.6,
    }

    def fake_evaluate(features, targets, splits, *, alphas, first_valid):
        coupling_seed = int(features[0, 0])
        model_code = int(features[0, 1])
        return pd.DataFrame(
            [
                {
                    "target": f"leg_tau{tau}_p1",
                    "capacity_test": capacities[(coupling_seed, model_code, tau)],
                    "alpha": 0.0,
                    "feature_dim": 1,
                }
                for tau in (1, 6)
            ]
        )

    monkeypatch.setattr(
        study.pipeline,
        "evaluate_feature_matrix",
        fake_evaluate,
    )
    config = {
        "scope": {"max_window": 5},
        "task_link": {
            "taus": [1, 6],
            "short_taus": [1],
            "long_taus": [6],
            "rows": 12,
            "washout": 1,
            "split_fractions": [0.5, 0.25, 0.25],
            "alpha_grid": [0.0],
            "bootstrap": 128,
            "bootstrap_seed": 123,
        },
        "acceptance_gates": {"positive_crossover_fraction": 0.75},
    }
    blocks = [
        {
            "physics_id": "p11",
            "coupling_seed": 11,
            "task_input_seed": 1011,
            "n_qubits": 2,
            "dt": 0.5,
            "gamma": 0.1,
            "dissipator": "local_loss",
        },
        {
            "physics_id": "p22",
            "coupling_seed": 22,
            "task_input_seed": 1022,
            "n_qubits": 2,
            "dt": 0.5,
            "gamma": 0.2,
            "dissipator": "local_loss",
        },
    ]

    scores, systems, summary = study.run_task_link(config, blocks)

    assert len(scores) == 8
    by_physics = systems.set_index("physics_id")
    assert by_physics.loc["p11", "short_delta"] == pytest.approx(-0.2)
    assert by_physics.loc["p11", "long_delta"] == pytest.approx(0.3)
    assert by_physics.loc["p11", "crossover_shift"] == pytest.approx(0.5)
    assert by_physics.loc["p22", "crossover_shift"] == pytest.approx(0.2)
    assert summary["mean_short_delta"] == pytest.approx(-0.15)
    assert summary["mean_long_delta"] == pytest.approx(0.2)
    assert summary["mean_crossover_shift"] == pytest.approx(0.35)
    assert summary["positive_shift_fraction"] == pytest.approx(1.0)
    assert summary["mean_crossover_shift_ci_low"] > 0.0
    assert summary["shift_gate"] is True
