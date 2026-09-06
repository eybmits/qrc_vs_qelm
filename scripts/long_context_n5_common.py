"""Numerical-material freeze and evidence checks for the separate N5 follow-up."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yaml

from qrc_vs_qelm.canonical_study import load_canonical_profile
from dense_phase_n4_common import prior_seed_registry, sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/canonical_emulation_long_n5.yaml"
FREEZE = ROOT / "configs/canonical_emulation_long_n5_freeze.json"
STUDY = "canonical_qrc_long_context_n5"
PROFILE = "long_n5"
SMOKE_PROFILE = "smoke_long_n5"
ROOTS = tuple(f"long_n5_{index:02d}" for index in range(1, 13))
GAMMAS = (1.0, 4.0)
WINDOWS = (2, 4, 6, 8, 10)
BUDGETS = (30, 60, 120)
SCIENCE_FILES = (
    "stream_metrics.csv", "suffix_decomposition.csv",
    "system_diagnostics.csv", "canonical_summary.json",
)
TEST_FILES = (
    "pra_extension/tests/test_long_context_n5.py",
    "pra_extension/tests/test_long_context_qelm_backend.py",
    "tests/test_canonical_study.py", "tests/test_canonical_emulation.py",
    "tests/test_canonical_resources.py", "tests/test_qelm.py", "tests/test_emulation.py",
)
NEW_MATERIAL = (
    "LONG_CONTEXT_N5_PROTOCOL.md", "scripts/long_context_n5_common.py",
    "scripts/long_context_qelm_backend.py", "scripts/run_long_context_n5.py",
    "scripts/analyze_long_context_n5.py", "scripts/dense_phase_n4_common.py",
    "pyproject.toml", *TEST_FILES,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def configuration(config: Path = CONFIG) -> dict[str, Any]:
    profile = load_canonical_profile(config, PROFILE)
    raw = yaml.safe_load(Path(config).read_text())
    if profile["study"] != STUDY or profile["status"] != "PROSPECTIVE_LONG_CONTEXT_N5":
        raise ValueError("N5 follow-up identity/status drift")
    if profile["confirmatory"] or tuple(str(r["root_id"]) for r in profile["roots"]) != ROOTS:
        raise ValueError("N5 follow-up must retain twelve separate prospective roots")
    if profile["teacher_n_qubits"] != [5] or profile["teacher_dissipators"] != ["local_loss"]:
        raise ValueError("N5 target family drift")
    if tuple(profile["teacher_gammas"]) != GAMMAS or tuple(profile["windows"]) != WINDOWS:
        raise ValueError("N5 loss/window grid drift")
    if profile["rows"] != {"train": 480, "val": 160, "test": 320}:
        raise ValueError("N5 split sizes drift")
    if (profile["suffix_groups"], profile["prefix_branches"], profile["prefix_burnin"]) != (32, 4, 128):
        raise ValueError("N5 suffix panel drift")
    student = profile["student"]
    if (student["n_qubits"], tuple(profile["snapshot_budgets"]), profile["max_snapshots"]) != (10, (1, 2, 4), 4):
        raise ValueError("N5 full-window emulator width/snapshot drift")
    if student["observables"] != {"axes": ["x", "y", "z"], "max_weight": 1}:
        raise ValueError("N5 single-Pauli emulator feature contract drift")
    if profile["teacher"]["target_observables"] != {"axes": ["x", "y", "z"], "max_weight": 2}:
        raise ValueError("N5 target observable contract drift")
    if profile["quantize_levels"] != 33 or profile["tolerance_nrmse"] != 0.10:
        raise ValueError("N5 alphabet or tolerance drift")
    if profile["decision"]["min_root_successes"] != 10 or profile["min_target_variance"] != 1e-8:
        raise ValueError("N5 robustness/liveness cutoff drift")
    if raw["analysis_seed"] != 96_990_001:
        raise ValueError("N5 bootstrap seed drift")
    for index, root in enumerate(profile["roots"], 1):
        observed = [root["teacher_coupling_seeds"]["5"], *[
            root[name] for name in (
                "student_coupling_seed", "train_input_seed", "val_input_seed", "test_input_seed",
                "suffix_input_seed", "prefix_input_seed", "classical_feature_seed",
            )
        ]]
        if observed != [96_000_000 + role * 10_000 + index for role in range(1, 9)]:
            raise ValueError("N5 role-seed namespace drift")
    return profile


def material_manifest(config: Path = CONFIG) -> dict[str, Any]:
    paths = {Path(config).resolve(), *[ROOT / name for name in NEW_MATERIAL]}
    paths.update((ROOT / "src/qrc_vs_qelm").rglob("*.py"))
    files = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(paths)}
    return {"files": files, "identity_sha256": sha256_json(files)}


def seed_audit(config: Path = CONFIG, reference_files=None) -> dict[str, Any]:
    raw = yaml.safe_load(Path(config).read_text())
    all_seeds = {int(raw["analysis_seed"]): "analysis_seed"}
    for profile_name, profile in raw["profiles"].items():
        for root in profile["roots"]:
            values = {f"teacher[{n}]": seed for n, seed in root["teacher_coupling_seeds"].items()}
            values.update({key: value for key, value in root.items() if key.endswith("_seed")})
            for role, seed in values.items():
                if int(seed) in all_seeds:
                    raise ValueError("duplicate development/follow-up/analysis seed")
                all_seeds[int(seed)] = f"{profile_name}.{root['root_id']}.{role}"
    registry = prior_seed_registry(config, reference_files)
    collisions = sorted(set(all_seeds) & {int(value) for value in registry["entries"]})
    if collisions:
        raise ValueError(f"follow-up seeds collide with prior structured records: {collisions}")
    return {
        "passed": True, "seed_count": len(all_seeds), "seed_map_sha256": sha256_json(all_seeds),
        "prior_seed_registry_sha256": registry["identity_sha256"],
        "prior_seed_source_files": registry["source_files"],
    }


def test_evidence(junit: Path) -> dict[str, Any]:
    root = ET.parse(junit).getroot()
    cases = root.findall(".//testcase")
    if not cases or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
        raise ValueError("test evidence must contain successful, unskipped cases")
    classnames = " ".join(case.get("classname", "") for case in cases)
    for name in TEST_FILES:
        if Path(name).stem not in classnames:
            raise ValueError(f"test evidence missing required suite {name}")
    return {
        "schema_version": 1, "passed": True, "case_count": len(cases),
        "junit_sha256": sha256_file(junit),
        "case_inventory": sorted((case.get("classname", ""), case.get("name", "")) for case in cases),
    }


def run_bound_tests(junit: Path, output: Path, config: Path = CONFIG) -> dict[str, Any]:
    """Execute the complete fixed suite while binding both sides to material."""
    if junit.exists() or output.exists():
        raise ValueError("refusing to overwrite existing bound test evidence")
    material = material_manifest(config)
    environment = dict(os.environ)
    environment["PYTEST_ADDOPTS"] = ""
    base = [sys.executable, "-m", "pytest", "-q", *TEST_FILES]
    collected = subprocess.run([*base, "--collect-only"], cwd=ROOT, env=environment, capture_output=True, text=True, check=True)
    inventory = sorted(line.strip() for line in collected.stdout.splitlines() if ".py::" in line and not line.startswith(" "))
    if not inventory:
        raise ValueError("complete numerical test inventory is empty")
    command = [*base, f"--junitxml={junit.resolve()}"]
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    print(completed.stdout, end="", flush=True)
    if completed.returncode:
        print(completed.stderr, end="", flush=True)
        raise ValueError("bound numerical test execution failed")
    if material_manifest(config) != material:
        raise ValueError("numerical material changed during bound test execution")
    evidence = test_evidence(junit)
    if len(inventory) != evidence["case_count"]:
        raise ValueError("JUnit did not execute the complete collected suite")
    actual = set(tuple(case) for case in evidence["case_inventory"])
    for node in inventory:
        filename, *names = node.split("::")
        class_suffix = ".".join([Path(filename).stem, *names[:-1]])
        if not any(classname.endswith(class_suffix) and name == names[-1] for classname, name in actual):
            raise ValueError(f"JUnit lacks collected case: {node}")
    result = {**evidence, "material": material, "executed_command": command,
              "collected_inventory": inventory, "junit_path": str(junit.resolve()),
              "material_unchanged_before_after": True}
    write_json(output, result)
    return result


def validate_test_evidence(evidence: dict[str, Any], material: dict[str, Any]) -> None:
    if evidence.get("material") != material or evidence.get("material_unchanged_before_after") is not True:
        raise ValueError("test evidence is not bound to executed unchanged numerical material")
    junit = Path(evidence.get("junit_path", ""))
    parsed = test_evidence(junit)
    for key, value in parsed.items():
        observed = evidence.get(key)
        if key == "case_inventory":
            observed = [tuple(item) for item in observed or []]
        if observed != value:
            raise ValueError("raw JUnit evidence differs from recorded test execution")
    command = evidence.get("executed_command", [])
    if command[1:] != ["-m", "pytest", "-q", *TEST_FILES, f"--junitxml={junit.resolve()}"]:
        raise ValueError("test execution was not the fixed complete suite")
    inventory = evidence.get("collected_inventory", [])
    if len(inventory) != parsed["case_count"] or len(set(inventory)) != len(inventory):
        raise ValueError("test inventory is incomplete or duplicated")
    actual = set(parsed["case_inventory"])
    for node in inventory:
        filename, *names = node.split("::")
        class_suffix = ".".join([Path(filename).stem, *names[:-1]])
        if not any(classname.endswith(class_suffix) and name == names[-1] for classname, name in actual):
            raise ValueError("raw JUnit omits collected test")


def validate_smoke(directory: Path, material: dict[str, Any]) -> dict[str, str]:
    audit = read_json(directory / "long_context_run_audit.json")
    if audit["profile"] != SMOKE_PROFILE or audit["material"] != material:
        raise ValueError("smoke is not bound to current development material")
    summary = read_json(directory / "canonical_summary.json")
    if summary["counts"] != {"roots": 1, "systems": 2, "stream_rows": 40, "suffix_rows": 40, "target_live_systems": 2}:
        raise ValueError("smoke coverage or target liveness failed")
    suffix = pd.read_csv(directory / "suffix_decomposition.csv")
    if suffix["identity_error"].abs().max() > 1e-10:
        raise ValueError("smoke decomposition failed")
    for column in ("predictions_agree", "suffix_bounds_satisfied"):
        if not suffix[column].eq(True).all():
            raise ValueError(f"smoke failed {column}")
    hashes = {name: sha256_file(directory / name) for name in SCIENCE_FILES}
    if hashes != audit["science_hashes"]:
        raise ValueError("smoke science artifact hash mismatch")
    return hashes


def create_freeze(smoke_a: Path, smoke_b: Path, evidence_path: Path, config: Path = CONFIG, freeze: Path = FREEZE) -> dict[str, Any]:
    if freeze.exists():
        raise ValueError("refusing to overwrite an existing N5 freeze")
    configuration(config)
    material = material_manifest(config)
    seeds = seed_audit(config)
    evidence = read_json(evidence_path)
    validate_test_evidence(evidence, material)
    if smoke_a.resolve() == smoke_b.resolve():
        raise ValueError("duplicate smoke evidence requires two distinct execution directories")
    execution_ids = [read_json(directory / "long_context_run_audit.json").get("execution_id") for directory in (smoke_a, smoke_b)]
    if not all(execution_ids) or len(set(execution_ids)) != 2:
        raise ValueError("duplicate smoke evidence requires two distinct executions")
    first = validate_smoke(smoke_a, material)
    second = validate_smoke(smoke_b, material)
    if first != second:
        raise ValueError("duplicated smoke science hashes differ")
    record = {
        "schema_version": 1, "study": STUDY, "profile": PROFILE,
        "status": "FROZEN_PROSPECTIVE_LONG_N5", "fresh_results_previously_opened": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "material": material, "seeds": seeds, "test_evidence": evidence,
        "test_evidence_sha256": sha256_file(evidence_path),
        "smoke_science_hashes": first,
        "smoke_directories": [str(smoke_a.resolve()), str(smoke_b.resolve())],
        "smoke_execution_ids": execution_ids,
    }
    write_json(freeze, record)
    return record


def verify_freeze(config: Path = CONFIG, freeze: Path = FREEZE) -> dict[str, Any]:
    configuration(config)
    record = read_json(freeze)
    material = material_manifest(config)
    if record.get("status") != "FROZEN_PROSPECTIVE_LONG_N5" or record.get("profile") != PROFILE:
        raise ValueError("N5 freeze identity/status invalid")
    if record.get("study") != STUDY or record.get("fresh_results_previously_opened") is not False:
        raise ValueError("N5 freeze lacks the prospective-opening attestation")
    if record.get("material") != material or record["test_evidence"].get("material") != material:
        raise ValueError("N5 frozen numerical material changed")
    validate_test_evidence(record["test_evidence"], material)
    seeds = seed_audit(config, record["seeds"]["prior_seed_source_files"])
    if seeds != record["seeds"]:
        raise ValueError("N5 prior-seed evidence changed")
    directories = record["smoke_directories"]
    if len(directories) != 2 or len({Path(path).resolve() for path in directories}) != 2:
        raise ValueError("N5 freeze lacks two distinct smoke directories")
    execution_ids = [read_json(Path(path) / "long_context_run_audit.json").get("execution_id") for path in directories]
    if len(set(execution_ids)) != 2 or execution_ids != record.get("smoke_execution_ids"):
        raise ValueError("N5 smoke execution identities changed")
    for directory in directories:
        if validate_smoke(Path(directory), material) != record["smoke_science_hashes"]:
            raise ValueError("N5 smoke evidence changed")
    return {"passed": True, "freeze_sha256": sha256_file(freeze), "material_identity_sha256": material["identity_sha256"]}


def median_bootstrap(values, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError("contrast must contain all twelve finite independent roots")
    rng = np.random.default_rng(int(seed))
    medians = np.median(values[rng.integers(0, 12, size=(10_000, 12))], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return {"median": float(np.median(values)), "ci95_low": float(low), "ci95_high": float(high),
            "positive_roots": int((values > 0).sum()), "n_roots": 12,
            "bootstrap_draws": 10_000, "bootstrap_unit": "root", "bootstrap_seed": int(seed)}
