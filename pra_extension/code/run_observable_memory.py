#!/usr/bin/env python3
"""Run the standalone PRA observable-memory stress test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observable_memory import (  # noqa: E402
    compact_findings,
    default_config,
    fit_decay_curves,
    run_stress_test,
    summarize_results,
)
from qrc_vs_qelm.provenance import (  # noqa: E402
    material_file_manifest,
    sha256_file,
    verify_canonical_freeze_record,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze_audit() -> dict[str, object]:
    config_path = REPO_ROOT / "configs" / "canonical_emulation.yaml"
    record_path = REPO_ROOT / "configs" / "canonical_emulation_freeze.json"
    material = material_file_manifest(
        repo_root=REPO_ROOT,
        config_path=config_path,
        runner_path=REPO_ROOT / "scripts" / "run_canonical_emulation.py",
    )
    audit = verify_canonical_freeze_record(
        record_path,
        study="canonical_qrc_to_parallel_qelm_emulation",
        profile="confirmatory",
        config_sha256=sha256_file(config_path),
        material_identity_sha256=str(material["identity_sha256"]),
    )
    return {
        "passed": bool(audit["passed"]),
        "material_file_count": int(material["file_count"]),
        "material_identity_sha256": str(material["identity_sha256"]),
        "freeze_record_sha256": audit.get("sha256"),
        "checks": audit.get("checks"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("smoke", "full", "continuous"), default="full"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "pra_extension" / "outputs" / "observable_memory_full",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")

    before = _freeze_audit()
    if not before["passed"]:
        raise RuntimeError("canonical freeze did not verify before extension run")
    config = default_config(args.profile)
    started = time.perf_counter()
    results = run_stress_test(config)
    decay = fit_decay_curves(results)
    summary = summarize_results(results, decay, config)
    summary["runtime_seconds"] = float(time.perf_counter() - started)
    summary["canonical_freeze_before"] = before

    results_path = output_dir / "observable_memory_results.csv"
    decay_path = output_dir / "observable_memory_decay_fits.csv"
    summary_path = output_dir / "observable_memory_summary.json"
    findings_path = output_dir / "observable_memory_findings.csv"
    results.to_csv(results_path, index=False)
    decay.to_csv(decay_path, index=False)
    pd.DataFrame(compact_findings(summary)).to_csv(findings_path, index=False)

    after = _freeze_audit()
    summary["canonical_freeze_after"] = after
    if not after["passed"] or (
        after["material_identity_sha256"] != before["material_identity_sha256"]
    ):
        raise RuntimeError("canonical material identity changed during extension run")
    summary["artifact_sha256"] = {
        results_path.name: _sha256(results_path),
        decay_path.name: _sha256(decay_path),
        findings_path.name: _sha256(findings_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "rows": int(len(results)),
        "decay_fits": int(len(decay)),
        "runtime_seconds": summary["runtime_seconds"],
        "max_transfer_prediction_error": summary["numerical_audit"]["max_transfer_prediction_error"],
        "canonical_freeze_passed": after["passed"],
        "material_identity_sha256": after["material_identity_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
