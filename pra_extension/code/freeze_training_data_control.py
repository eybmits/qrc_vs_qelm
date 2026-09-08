"""Validate engineering evidence and write a non-overwriting local follow-up freeze.

This is a local provenance record, not public preregistration or scientific
evidence. It does not run experiments or open production roots. Two distinct
development smoke runs, a development benchmark, and passing JUnit evidence
must match the current configuration and frozen source material.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "src", ROOT / "scripts", Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import dense_phase_n4_common as historical_seeds  # noqa: E402
import run_training_data_control as runner  # noqa: E402

REQUIRED_ARRAYS = {
    "reference_windows_chronological", "panel_windows_chronological", "panel_prefix_inputs",
    "mean_prefix_state", "reference_conditional_means", "long_mean_prefix_state",
    "long_reference_conditional_means", "panel_exact_means", "panel_targets",
    *(f"{kind}_{split}" for kind in ("input", "target") for split in ("train", "val", "test")),
    *(f"{kind}_{split}" for kind in ("qelm_max_features", "poly2_features")
      for split in ("train", "val", "test", "reference", "panel")),
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read_json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value in {path}: {value}")

    value = json.loads(path.read_text(), object_pairs_hook=unique,
                       parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch")


def _hashes(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a nonempty hash mapping")
    if any(not isinstance(key, str) or not isinstance(digest, str)
           or not re.fullmatch(r"[0-9a-f]{64}", digest) for key, digest in value.items()):
        raise ValueError(f"{label} contains an invalid SHA256")


def _without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_runtime(item) for key, item in value.items()
                if key not in {"runtime_seconds", "timing_seconds"}}
    if isinstance(value, list):
        return [_without_runtime(item) for item in value]
    return value


def validate_junit(path: Path) -> dict[str, Any]:
    """Require actual passing test cases, not merely a successful-looking total."""
    tree = ET.parse(path).getroot()
    if tree.tag not in {"testsuite", "testsuites"}:
        raise ValueError("JUnit root must be testsuite or testsuites")
    cases = list(tree.iter("testcase"))
    if not cases:
        raise ValueError("JUnit has no test cases")
    for tag in ("failure", "error", "skipped"):
        if any(True for _ in tree.iter(tag)):
            raise ValueError(f"JUnit contains {tag}: all tests must pass")
    for suite in [tree, *tree.iter("testsuite")]:
        for field in ("failures", "errors", "skipped", "disabled"):
            try:
                count = int(suite.attrib.get(field, "0"))
            except ValueError as exc:
                raise ValueError(f"JUnit has invalid {field} count") from exc
            if count != 0:
                raise ValueError(f"JUnit reports nonpassing {field}")
        if "tests" in suite.attrib:
            try:
                count = int(suite.attrib["tests"])
            except ValueError as exc:
                raise ValueError("JUnit has invalid test count") from exc
            if count != len(list(suite.iter("testcase"))):
                raise ValueError("JUnit test count does not match test cases")
    return {"passed": True, "tests": len(cases),
            "path": str(path.resolve()), "sha256": runner.config_file_sha256(path)}


def validate_run(path: Path, profile: str, config: dict[str, Any],
                 config_sha: str, material_sha: str) -> dict[str, Any]:
    """Check the manifest-to-CSV chain and the seeds actually executed."""
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"run directory is missing: {path}")
    if list(path.rglob("failure.json")):
        raise ValueError(f"run contains failure evidence: {path}")
    manifest = _read_json(path / "manifest.json")
    cfg = runner.profile_config(config, profile)
    ledger = runner.seed_manifest(config)
    root_ids = [root["root_id"] for root in cfg["roots"]]
    for key, expected in {
        "study": config["study"], "profile": profile, "status": "COMPLETED",
        "scientific_evidence": False, "config_sha256": config_sha,
        "material_sha256": material_sha, "effective_config": cfg,
        "selected_root_ids": root_ids, "seed_manifest": ledger,
        "production_freeze": None,
    }.items():
        if key not in manifest:
            raise ValueError(f"{profile} manifest missing {key}")
        _equal(manifest[key], expected, f"{profile} {key}")
    bound_material = manifest.get("material_manifest")
    if not isinstance(bound_material, dict):
        raise ValueError(f"{profile} material_manifest missing")
    _hashes(bound_material.get("files"), f"{profile} material files")
    _equal(bound_material.get("identity_sha256"), material_sha, f"{profile} material identity")
    _equal(sha256(_json_bytes(bound_material["files"])).hexdigest(), material_sha,
           f"{profile} material file map")
    runtime = manifest.get("runtime_seconds")
    if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or not np.isfinite(runtime) or runtime <= 0:
        raise ValueError(f"{profile} runtime must be finite and positive")
    records = manifest.get("roots")
    if not isinstance(records, list) or [item.get("root_id") for item in records] != root_ids:
        raise ValueError(f"{profile} root records mismatch")

    scientific_roots = []
    root_tables = []
    for root, record in zip(cfg["roots"], records):
        root_dir = path / root["root_id"]
        root_path = root_dir / "manifest.json"
        _equal(record.get("manifest_sha256"), runner.config_file_sha256(root_path),
               f"{profile} root manifest hash")
        root_manifest = _read_json(root_path)
        for key, expected in {"study": config["study"], "profile": profile,
                              "root_id": root["root_id"], "config_sha256": config_sha,
                              "material_sha256": material_sha}.items():
            _equal(root_manifest.get(key), expected, f"{profile} root {key}")
        csv_path = root_dir / "metrics.csv"
        _equal(root_manifest.get("metrics_sha256"), runner.config_file_sha256(csv_path),
               f"{profile} root metrics hash")
        table = pd.read_csv(csv_path)
        if not (table["profile"] == profile).all():
            raise ValueError(f"{profile} CSV profile mismatch")
        coverage = runner.validate_coverage(table, cfg, [root])
        _equal(root_manifest.get("coverage"), coverage, f"{profile} root coverage")
        root_tables.append(table)
        anchors = root_manifest.get("anchors")
        if not isinstance(anchors, list) or len(anchors) != len(cfg["anchors"]):
            raise ValueError(f"{profile} anchor records mismatch")
        for anchor, item in zip(cfg["anchors"], anchors):
            _equal(item.get("anchor"), anchor, f"{profile} anchor")
            executed_seeds = [seed for seed in ledger if seed["profile"] == profile
                              and seed["root_id"] == root["root_id"]
                              and seed["anchor_id"] == anchor["id"]]
            _equal(item.get("seeds"), executed_seeds, f"{profile} executed seeds")
            _hashes(item.get("array_sha256"), f"{profile} array hashes")
            _equal(set(item["array_sha256"]), REQUIRED_ARRAYS, f"{profile} array hash inventory")
            _hashes(item.get("train_slice_input_sha256"), f"{profile} training-slice hashes")
            _equal(set(item["train_slice_input_sha256"]),
                   {str(count) for count in cfg["training_rows"]}, f"{profile} training slices")
        scientific_roots.append({"root_id": root["root_id"],
                                 "metrics_sha256": root_manifest["metrics_sha256"],
                                 "anchors": _without_runtime(anchors)})

    _equal(manifest.get("metrics_sha256"), runner.config_file_sha256(path / "metrics.csv"),
           f"{profile} combined metrics hash")
    combined = pd.read_csv(path / "metrics.csv")
    if not (combined["profile"] == profile).all():
        raise ValueError(f"{profile} combined CSV profile mismatch")
    coverage = runner.validate_coverage(combined, cfg, cfg["roots"])
    _equal(manifest.get("coverage"), coverage, f"{profile} combined coverage")
    pd.testing.assert_frame_equal(combined, pd.concat(root_tables, ignore_index=True))
    artifacts = {}
    for artifact in sorted(path.rglob("*")):
        if artifact.is_symlink():
            raise ValueError(f"run artifacts must not be symlinks: {artifact}")
        if artifact.is_file():
            artifacts[artifact.relative_to(path).as_posix()] = runner.config_file_sha256(artifact)
    scientific = {"metrics_sha256": manifest["metrics_sha256"], "roots": scientific_roots}
    return {"path": str(path), "profile": profile, "artifacts": artifacts,
            "scientific_sha256": sha256(_json_bytes(scientific)).hexdigest(),
            "scientific": scientific, "runtime_seconds": runtime, "coverage": coverage}


def audit_unopened_roots(config: dict[str, Any]) -> dict[str, Any]:
    """Audit known follow-up result directories, including partial/failed runs.

    A top-level seed_manifest documents unopened seeds too; it is deliberately
    not treated as proof that those roots executed. Execution fields and
    production root directories are audited instead.
    """
    production_ids = {root["root_id"] for root in config["roots"]}
    production_seeds = {item["seed"] for item in runner.seed_manifest(config)
                        if item["profile"] == "production"}
    files = {}
    searched = []
    for folder in sorted(ROOT.glob("results_training_data_control_*")):
        if not folder.is_dir():
            continue
        searched.append(str(folder.resolve()))
        for item in folder.rglob("*"):
            if item.is_dir() and item.name in production_ids:
                raise ValueError(f"production root previously opened: {item}")
        manifests = sorted({*folder.rglob("manifest.json"), *folder.rglob("failure.json")})
        for path in manifests:
            value = _read_json(path)
            files[str(path.relative_to(ROOT))] = runner.config_file_sha256(path)
            actual_ids = {value.get("root_id")}
            actual_ids.update(value.get("selected_root_ids", []))
            actual_ids.update(item.get("root_id") for item in value.get("completed_roots", []))
            actual_ids.update(item.get("root_id") for item in value.get("effective_config", {}).get("roots", []))
            executed = [seed for anchor in value.get("anchors", [])
                        for seed in anchor.get("seeds", [])]
            if (value.get("profile") == "production" or actual_ids & production_ids
                    or any(seed.get("profile") == "production"
                           or seed.get("seed") in production_seeds for seed in executed)):
                raise ValueError(f"production root previously opened: {path}")
    return {"passed": True, "scope": "repository results_training_data_control_* directories",
            "directories": searched, "artifact_sha256": files}


def audit_seeds(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    ledger = runner.seed_manifest(config)
    candidates = [(root["seed"], f"root:{root['root_id']}") for root in config["roots"]]
    candidates.append((config["development_seed"], "development_seed"))
    if "analysis_seed" in config:
        candidates.append((config["analysis_seed"], "analysis_seed"))
    base_seeds = [{"seed": seed, "label": label} for seed, label in candidates]
    candidates.extend((item["seed"], f"{item['profile']}:{item['root_id']}:{item['anchor_id']}:{item['role']}")
                      for item in ledger)
    seen = {}
    for seed, label in candidates:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"invalid seed: {label}")
        if seed in seen:
            raise ValueError(f"seed collision: {label} and {seen[seed]}")
        seen[seed] = label
    prior = historical_seeds.prior_seed_registry(config_path)
    conflicts = [{"seed": seed, "label": label,
                  "prior_locations": prior["entries"][str(seed)]}
                 for seed, label in candidates if str(seed) in prior["entries"]]
    if conflicts:
        raise ValueError(f"historical seed collisions: {conflicts}")
    return {"passed": True, "seed_count": len(candidates), "seed_manifest": ledger,
            "base_seeds": base_seeds,
            "prior_seed_count": prior["seed_count"],
            "prior_seed_registry_sha256": prior["identity_sha256"],
            "prior_seed_source_files": prior["source_files"]}


def dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "pandas", "pytest", "PyYAML"):
        versions[name] = metadata.version(name)
    return versions


def create_freeze(config_path: Path, smoke_a: Path, smoke_b: Path, benchmark: Path,
                  tests_junit: Path, output: Path) -> dict[str, Any]:
    """Validate all inputs before exclusively creating the requested JSON file."""
    config_path, output = Path(config_path).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite freeze output: {output}")
    run_paths = [Path(path).resolve() for path in (smoke_a, smoke_b, benchmark)]
    if len(set(run_paths)) != 3:
        raise ValueError("two smoke runs and benchmark must use distinct directories")
    if any(output.resolve().is_relative_to(path) for path in run_paths):
        raise ValueError("freeze output must not be inside an input run directory")
    config = runner.read_config(config_path)
    config_sha = runner.config_file_sha256(config_path)
    material = runner.material_manifest(config_path)
    junit = validate_junit(Path(tests_junit))
    runs = [validate_run(path, profile, config, config_sha, material["identity_sha256"])
            for path, profile in zip(run_paths, ("smoke", "smoke", "benchmark"))]
    _equal(runs[0]["scientific_sha256"], runs[1]["scientific_sha256"],
           "smoke scientific CSV/array evidence")
    roots_audit = audit_unopened_roots(config)
    seeds_audit = audit_seeds(config_path, config)
    record = {
        "schema_version": 1, "status": "FROZEN_FOLLOWUP", "study": config["study"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registration_scope": "local record only; not public preregistration",
        "scientific_evidence": False,
        "config_sha256": config_sha, "material_sha256": material["identity_sha256"],
        "material_files": material["files"],
        "root_ids": [root["root_id"] for root in config["roots"]],
        "production_roots_previously_opened": False,
        "production_root_audit": roots_audit, "seed_audit": seeds_audit,
        "seed_manifest": seeds_audit["seed_manifest"],
        "dependencies": dependency_versions(), "tests_junit": junit,
        "engineering_runs": {label: {key: value for key, value in run.items() if key != "scientific"}
                             for label, run in zip(("smoke_a", "smoke_b", "benchmark"), runs)},
    }
    # Recheck bound source/config bytes before writing; x mode also closes the
    # output-existence race without replacing anyone else's freeze record.
    _equal(runner.config_file_sha256(config_path), config_sha, "configuration changed during audit")
    _equal(runner.material_manifest(config_path), material, "material changed during audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(_json_bytes(record) + b"\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training_data_control.json")
    parser.add_argument("--smoke-a", type=Path, required=True)
    parser.add_argument("--smoke-b", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--tests-junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/training_data_control_freeze.json")
    args = parser.parse_args()
    record = create_freeze(args.config, args.smoke_a, args.smoke_b, args.benchmark,
                           args.tests_junit, args.output)
    print(json.dumps({"status": record["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
