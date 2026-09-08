"""Synthetic artifact tests: no development or production experiment is run."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "pra_extension/code", REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import freeze_training_data_control as freeze  # noqa: E402
import run_training_data_control as runner  # noqa: E402


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def _read_json(path):
    return json.loads(path.read_text())


def _refresh_root_hash(directory):
    summary = _read_json(directory / "manifest.json")
    root_id = summary["selected_root_ids"][0]
    summary["roots"][0]["manifest_sha256"] = runner.config_file_sha256(
        directory / root_id / "manifest.json"
    )
    _write_json(directory / "manifest.json", summary)


def _synthetic_run(directory, config_path, profile, runtime):
    """Construct minimal internally consistent fake evidence for validation tests."""
    config = runner.read_config(config_path)
    cfg = runner.profile_config(config, profile)
    root = cfg["roots"][0]
    anchor = cfg["anchors"][0]
    root_dir = directory / root["root_id"]
    root_dir.mkdir(parents=True)
    digest = "a" * 64
    rows = [{"root_id": root["root_id"], "anchor_id": anchor["id"], "profile": profile,
             "training_rows": count, "model": model, "snapshot_budget": budget,
             "fit_rows": count + cfg["validation_rows"], "target_variance": 0.25,
             "reference_mean_sha256": digest, "test_mse": 0.1, "test_nrmse": 0.2}
            for count in cfg["training_rows"]
            for model, budget in (("ptd_parallel", 1), ("ptd_parallel", 2),
                                  ("ptd_parallel", 4), ("poly2_delay", 0))]
    table = pd.DataFrame(rows)
    table.to_csv(root_dir / "metrics.csv", index=False)
    table.to_csv(directory / "metrics.csv", index=False)
    coverage = runner.validate_coverage(table, cfg, [root])
    ledger = runner.seed_manifest(config)
    root_manifest = {
        "schema_version": 1, "study": config["study"], "profile": profile,
        "root_id": root["root_id"], "config_sha256": runner.config_file_sha256(config_path),
        "material_sha256": runner.material_manifest(config_path)["identity_sha256"],
        "metrics_sha256": runner.config_file_sha256(root_dir / "metrics.csv"),
        "coverage": coverage,
        "anchors": [{"anchor": anchor,
                     "seeds": [item for item in ledger if item["profile"] == profile],
                     "array_sha256": {key: digest for key in freeze.REQUIRED_ARRAYS},
                     "train_slice_input_sha256": {str(count): digest for count in cfg["training_rows"]},
                     "reference_prefix_length": cfg["prefix_length"],
                     "timing_seconds": {"total_seconds": runtime}, "rows": len(table)}],
    }
    _write_json(root_dir / "manifest.json", root_manifest)
    summary = {
        "schema_version": 1, "study": config["study"], "profile": profile,
        "status": "COMPLETED", "scientific_evidence": False,
        "config_sha256": runner.config_file_sha256(config_path),
        "material_sha256": runner.material_manifest(config_path)["identity_sha256"],
        "material_manifest": runner.material_manifest(config_path),
        "effective_config": cfg, "selected_root_ids": [root["root_id"]],
        "production_freeze": None, "seed_manifest": ledger, "coverage": coverage,
        "roots": [{"root_id": root["root_id"],
                   "manifest_sha256": runner.config_file_sha256(root_dir / "manifest.json")}],
        "metrics_sha256": runner.config_file_sha256(directory / "metrics.csv"),
        "runtime_seconds": runtime,
    }
    _write_json(directory / "manifest.json", summary)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # These fixtures never call runner.run or run_anchor and contain no
    # experimental measurements. All files, including test freezes, are temp.
    config = json.loads((REPO_ROOT / "configs/training_data_control.json").read_text())
    config["roots"] = config["roots"][:2]
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    source = tmp_path / "synthetic_material.txt"
    source.write_text("test fixture, not production material")

    def material(path):
        files = {"config.json": runner.config_file_sha256(path),
                 "synthetic_material.txt": runner.config_file_sha256(source)}
        identity = sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"files": files, "identity_sha256": identity}

    monkeypatch.setattr(runner, "material_manifest", material)
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    monkeypatch.setattr(freeze.historical_seeds, "prior_seed_registry", lambda path: {
        "entries": {}, "source_files": {}, "identity_sha256": "b" * 64, "seed_count": 0,
    })
    paths = [tmp_path / f"results_training_data_control_{name}"
             for name in ("smoke_a", "smoke_b", "benchmark")]
    for path, profile, runtime in zip(paths, ("smoke", "smoke", "benchmark"), (1.0, 2.0, 3.0)):
        _synthetic_run(path, config_path, profile, runtime)
    junit = tmp_path / "tests.xml"
    junit.write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
                     '<testcase classname="synthetic" name="passed"/>'
                     '</testsuite></testsuites>')
    return {"config_path": config_path, "smoke_a": paths[0], "smoke_b": paths[1],
            "benchmark": paths[2], "tests_junit": junit, "output": tmp_path / "freeze.json"}


def test_valid_local_freeze_is_bound_and_production_verifiable(evidence):
    result = freeze.create_freeze(**evidence)
    assert result["status"] == "FROZEN_FOLLOWUP"
    assert result["registration_scope"] == "local record only; not public preregistration"
    assert result["production_roots_previously_opened"] is False
    assert result["production_root_audit"]["passed"]
    assert result["seed_audit"]["passed"]
    assert result["created_utc"].endswith("+00:00")
    assert "numpy" in result["dependencies"]
    assert result["tests_junit"]["sha256"] == runner.config_file_sha256(evidence["tests_junit"])
    assert runner.verify_freeze(evidence["config_path"], evidence["output"]) == result
    first = result["engineering_runs"]["smoke_a"]
    second = result["engineering_runs"]["smoke_b"]
    assert first["scientific_sha256"] == second["scientific_sha256"]
    assert first["artifacts"]["manifest.json"] != second["artifacts"]["manifest.json"]
    assert len(first["artifacts"]) == 4


def test_existing_output_is_never_replaced(evidence):
    evidence["output"].write_text("preserve me")
    with pytest.raises(FileExistsError, match="overwrite"):
        freeze.create_freeze(**evidence)
    assert evidence["output"].read_text() == "preserve me"


def test_distinct_runs_required(evidence):
    evidence["smoke_b"] = evidence["smoke_a"]
    with pytest.raises(ValueError, match="distinct"):
        freeze.create_freeze(**evidence)


@pytest.mark.parametrize("xml", [
    '<testsuite tests="0"/>',
    '<testsuite tests="1" failures="1"><testcase name="x"/></testsuite>',
    '<testsuite tests="1"><testcase name="x"><failure/></testcase></testsuite>',
    '<testsuite tests="1"><testcase name="x"><error/></testcase></testsuite>',
    '<testsuite tests="1"><testcase name="x"><skipped/></testcase></testsuite>',
    '<testsuite tests="2"><testcase name="x"/></testsuite>',
    '<testsuite tests="1" errors="garbage"><testcase name="x"/></testsuite>',
])
def test_nonpassing_or_inconsistent_junit_is_rejected(evidence, xml):
    evidence["tests_junit"].write_text(xml)
    with pytest.raises(ValueError, match="JUnit"):
        freeze.create_freeze(**evidence)
    assert not evidence["output"].exists()


def test_csv_tampering_is_rejected(evidence):
    path = evidence["smoke_a"] / "development_smoke/metrics.csv"
    path.write_text(path.read_text().replace("0.1,0.2", "0.9,0.2"))
    with pytest.raises(ValueError, match="metrics hash"):
        freeze.create_freeze(**evidence)


def test_scientific_hash_difference_is_rejected_even_with_valid_manifest_chain(evidence):
    path = evidence["smoke_b"] / "development_smoke/manifest.json"
    value = _read_json(path)
    value["anchors"][0]["array_sha256"]["input_train"] = "c" * 64
    _write_json(path, value)
    _refresh_root_hash(evidence["smoke_b"])
    with pytest.raises(ValueError, match="smoke scientific"):
        freeze.create_freeze(**evidence)


@pytest.mark.parametrize("change", ["array_inventory", "root_material", "material_map"])
def test_incomplete_or_inconsistent_hash_inventory_is_rejected(evidence, change):
    directory = evidence["smoke_a"]
    path = directory / "development_smoke/manifest.json"
    value = _read_json(path)
    if change == "array_inventory":
        del value["anchors"][0]["array_sha256"]["panel_targets"]
        _write_json(path, value)
        _refresh_root_hash(directory)
    elif change == "root_material":
        value["material_sha256"] = "0" * 64
        _write_json(path, value)
        _refresh_root_hash(directory)
    else:
        path = directory / "manifest.json"
        value = _read_json(path)
        value["material_manifest"]["files"]["synthetic_material.txt"] = "0" * 64
        _write_json(path, value)
    with pytest.raises(ValueError, match="array hash inventory|material_sha256|material file map"):
        freeze.create_freeze(**evidence)


@pytest.mark.parametrize("run,key,value", [
    ("smoke_a", "profile", "production"),
    ("benchmark", "profile", "smoke"),
    ("smoke_a", "config_sha256", "0" * 64),
    ("benchmark", "material_sha256", "0" * 64),
    ("smoke_a", "status", "FAILED"),
    ("benchmark", "scientific_evidence", True),
    ("benchmark", "runtime_seconds", 0.0),
])
def test_wrong_run_contract_is_rejected(evidence, run, key, value):
    path = evidence[run] / "manifest.json"
    summary = _read_json(path)
    summary[key] = value
    _write_json(path, summary)
    with pytest.raises(ValueError, match=key if key != "runtime_seconds" else "runtime"):
        freeze.create_freeze(**evidence)


def test_development_profile_cannot_use_production_base_seed(evidence):
    path = evidence["benchmark"] / "manifest.json"
    summary = _read_json(path)
    summary["effective_config"]["roots"][0]["seed"] = runner.read_config(evidence["config_path"])["roots"][0]["seed"]
    _write_json(path, summary)
    with pytest.raises(ValueError, match="effective_config"):
        freeze.create_freeze(**evidence)


def test_execution_ledger_cannot_use_production_derived_seeds(evidence):
    path = evidence["smoke_a"] / "development_smoke/manifest.json"
    value = _read_json(path)
    value["anchors"][0]["seeds"][0] = runner.seed_manifest(runner.read_config(evidence["config_path"]))[0]
    _write_json(path, value)
    _refresh_root_hash(evidence["smoke_a"])
    with pytest.raises(ValueError, match="executed seeds"):
        freeze.create_freeze(**evidence)


def test_current_material_changes_invalidate_smokes(evidence):
    (evidence["config_path"].parent / "synthetic_material.txt").write_text("changed source")
    with pytest.raises(ValueError, match="material_sha256"):
        freeze.create_freeze(**evidence)


@pytest.mark.parametrize("partial", [False, True])
def test_prior_production_execution_blocks_freeze(evidence, partial):
    config = runner.read_config(evidence["config_path"])
    folder = evidence["config_path"].parent / "results_training_data_control_earlier"
    folder.mkdir()
    if partial:
        (folder / config["roots"][0]["root_id"]).mkdir()
    else:
        _write_json(folder / "failure.json", {"profile": "production", "status": "FAILED"})
    with pytest.raises(ValueError, match="previously opened"):
        freeze.create_freeze(**evidence)


@pytest.mark.parametrize("role", ["base", "derived", "analysis"])
def test_historical_seed_collisions_are_rejected(evidence, monkeypatch, role):
    config = runner.read_config(evidence["config_path"])
    seed = {"base": config["roots"][0]["seed"],
            "derived": runner.seed_manifest(config)[0]["seed"],
            "analysis": config["analysis_seed"]}[role]
    monkeypatch.setattr(freeze.historical_seeds, "prior_seed_registry", lambda path: {
        "entries": {str(seed): ["prior.json:seed"]}, "source_files": {"prior.json": "d" * 64},
        "identity_sha256": "e" * 64, "seed_count": 1,
    })
    with pytest.raises(ValueError, match="historical seed collisions"):
        freeze.create_freeze(**evidence)


def test_internal_analysis_seed_collision_is_rejected(evidence):
    config = deepcopy(runner.read_config(evidence["config_path"]))
    config["analysis_seed"] = config["development_seed"]
    with pytest.raises(ValueError, match="seed collision"):
        freeze.audit_seeds(evidence["config_path"], config)


def test_failure_artifact_blocks_even_completed_manifest(evidence):
    _write_json(evidence["smoke_a"] / "failure.json", {"status": "FAILED"})
    with pytest.raises(ValueError, match="failure evidence"):
        freeze.create_freeze(**evidence)


def test_output_cannot_change_an_input_directory(evidence):
    evidence["output"] = evidence["smoke_a"] / "freeze.json"
    with pytest.raises(ValueError, match="inside an input run"):
        freeze.create_freeze(**evidence)
