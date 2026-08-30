#!/usr/bin/env python
"""Evaluate frozen canonical gates without opening or rerunning study roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np

from qrc_vs_qelm.canonical_gates import (
    evaluate_canonical_gates,
    load_canonical_artifacts,
    verify_canonical_manifest,
    verify_duplicate_smoke_runs,
    verify_pytest_junit,
)
from qrc_vs_qelm.canonical_study import load_canonical_profile
from qrc_vs_qelm.provenance import sha256_file, verify_canonical_freeze_record


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "canonical_emulation.yaml"
REPORT_NAME = "canonical_gate_report.json"
CONTRASTS_NAME = "canonical_block_contrasts.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate G0-G6 and preregistered secondary diagnostics."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke", "calibration", "confirmatory"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pytest-junit",
        type=Path,
        help="JUnit XML from the complete pytest suite; absent evidence fails G0.",
    )
    parser.add_argument(
        "--smoke-runs",
        nargs=2,
        type=Path,
        metavar=("FIRST", "SECOND"),
        help="Two independently generated smoke output directories.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Report directory (defaults to the canonical results directory).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _collect_full_suite_count() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    matches = re.findall(r"(\d+)\s+tests?\s+collected", combined)
    if result.returncode != 0 or not matches:
        raise RuntimeError("could not verify the current full pytest collection")
    return int(matches[-1])


def main() -> int:
    args = parse_args()
    results = args.results.resolve()
    config = args.config.resolve()
    profile = load_canonical_profile(config, args.profile)
    stream, suffix, systems, summary, manifest = load_canonical_artifacts(results)
    if summary.get("schema_version") != 1:
        raise ValueError("canonical summary has an unsupported schema version")
    for label, observed in (
        ("study", summary.get("study")),
        ("profile", summary.get("profile")),
    ):
        if str(observed) != str(profile[label]):
            raise ValueError(
                f"canonical summary {label} {observed!r} disagrees with profile {profile[label]!r}"
            )
        if str(manifest.get(label)) != str(profile[label]):
            raise ValueError(
                f"canonical manifest {label} disagrees with profile {profile[label]!r}"
            )
    if bool(summary.get("confirmatory")) != bool(profile["confirmatory"]):
        raise ValueError("canonical summary confirmatory flag disagrees with profile")

    manifest_audit = verify_canonical_manifest(
        results,
        manifest,
        material_root=ROOT,
        config_path=config,
    )
    freeze_audit: dict[str, Any]
    if profile["confirmatory"]:
        material = manifest.get("reproducibility", {}).get("material", {})
        freeze_audit = verify_canonical_freeze_record(
            Path(profile["freeze_record_path"]),
            study=str(profile["study"]),
            profile=str(profile["profile"]),
            config_sha256=str(manifest.get("config_sha256")),
            material_identity_sha256=str(material.get("identity_sha256")),
        )
        run_freeze = manifest.get("freeze_record_audit")
        run_record_matches = bool(
            isinstance(run_freeze, dict)
            and run_freeze.get("passed") is True
            and run_freeze.get("sha256") == freeze_audit.get("sha256")
        )
        freeze_audit["run_record_matches_current"] = run_record_matches
        freeze_audit["passed"] = bool(
            freeze_audit.get("passed") and run_record_matches
        )
    else:
        freeze_audit = {
            "passed": True,
            "reason": "nonconfirmatory profile does not require unseal approval",
        }
    tests_audit: dict[str, Any] = {
        "passed": False,
        "reason": "--pytest-junit was not supplied",
    }
    if args.pytest_junit is not None:
        expected_material_identity = str(
            manifest.get("reproducibility", {})
            .get("material", {})
            .get("identity_sha256", "")
        )
        tests_audit = verify_pytest_junit(
            args.pytest_junit.resolve(),
            expected_testcases=_collect_full_suite_count(),
            expected_material_identity_sha256=expected_material_identity,
            expected_config_sha256=str(manifest.get("config_sha256", "")),
        )
    smoke_audit: dict[str, Any] = {
        "passed": False,
        "reason": "--smoke-runs was not supplied",
    }
    if args.smoke_runs is not None:
        smoke_audit = verify_duplicate_smoke_runs(
            args.smoke_runs[0].resolve(),
            args.smoke_runs[1].resolve(),
            expected_material_identity_sha256=str(
                manifest.get("reproducibility", {})
                .get("material", {})
                .get("identity_sha256", "")
            ),
        )

    evidence = {
        "tests_passed": bool(tests_audit["passed"]),
        "strict_json": True,
        "manifest_hashes_valid": bool(manifest_audit["passed"]),
        "freeze_identity_matches": bool(freeze_audit["passed"]),
        "seed_namespaces_disjoint": True,
        "duplicate_smoke_deterministic": bool(smoke_audit["passed"]),
    }
    report, contrasts = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=evidence,
    )
    report["g0_audit"] = {
        "pytest_junit": tests_audit,
        "manifest": manifest_audit,
        "freeze_record": freeze_audit,
        "duplicate_smoke": smoke_audit,
        "seed_validation": {
            "passed": True,
            "scope": "all configured smoke, calibration, and confirmatory roots",
        },
        "strict_json": {
            "passed": True,
            "files": [
                "canonical_summary.json",
                "computational_resources.json",
                "canonical_manifest.json",
            ],
        },
    }
    report["evaluation_provenance"] = {
        "config_sha256": sha256_file(config),
        "evaluator_module_sha256": sha256_file(
            ROOT / "src" / "qrc_vs_qelm" / "canonical_gates.py"
        ),
        "input_artifact_sha256": {
            name: sha256_file(results / name)
            for name in (
                "stream_metrics.csv",
                "suffix_decomposition.csv",
                "system_diagnostics.csv",
                "canonical_summary.json",
                "computational_resources.json",
                "canonical_manifest.json",
            )
        },
    }

    output = (args.out or results).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / REPORT_NAME
    contrasts_path = output / CONTRASTS_NAME
    existing = [path for path in (report_path, contrasts_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite gate artifacts without --overwrite: {existing}"
        )
    contrasts.to_csv(
        contrasts_path,
        index=False,
        float_format="%.17g",
    )
    report["block_contrasts"] = {
        "path": CONTRASTS_NAME,
        "rows": int(len(contrasts)),
        "sha256": sha256_file(contrasts_path),
    }
    _write_json(report_path, report)
    print(json.dumps({"decision": report["decision"], "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
