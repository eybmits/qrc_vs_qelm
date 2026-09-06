#!/usr/bin/env python
"""Freeze or execute the separate N5 long-context amendment; old studies are untouched."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

from qrc_vs_qelm import canonical_study

from long_context_n5_common import (
    CONFIG, FREEZE, PROFILE, SCIENCE_FILES, SMOKE_PROFILE, configuration,
    create_freeze, material_manifest, run_bound_tests, sha256_file, verify_freeze, write_json,
)
from long_context_qelm_backend import efficient_qelm_backend


def run(profile: str, output: Path, config: Path = CONFIG) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output: {output}")
    if profile not in (SMOKE_PROFILE, PROFILE):
        raise ValueError("unknown follow-up profile")
    configuration(config)
    freeze_audit = verify_freeze(config) if profile == PROFILE else None
    material = material_manifest(config)
    original_inputs = canonical_study._input_blocks
    original_reservoir = canonical_study._teacher_reservoir
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    execution_id = str(uuid4())

    def logged_inputs(root, settings):
        print(f"root_start {root['root_id']} elapsed_seconds={perf_counter()-started:.1f}", flush=True)
        return original_inputs(root, settings)

    def logged_reservoir(settings, **kwargs):
        print(f"target_start N={kwargs['n_qubits']} gamma={kwargs['gamma']} elapsed_seconds={perf_counter()-started:.1f}", flush=True)
        return original_reservoir(settings, **kwargs)

    with efficient_qelm_backend(), patch.object(canonical_study, "_input_blocks", logged_inputs), patch.object(canonical_study, "_teacher_reservoir", logged_reservoir):
        summary = canonical_study.run_canonical_study(
            config_path=config, profile_name=profile, output_dir=output, runner_path=Path(__file__).resolve(),
        )
    if material_manifest(config) != material:
        raise ValueError("numerical source changed during execution; result cannot be accepted")
    audit = {
        "schema_version": 1, "profile": profile, "material": material, "freeze": freeze_audit,
        "execution_id": execution_id, "started_at_utc": started_at,
        "backend": "identical_dense_unitaries_direct_product_encoding_sparse_local_pauli_expectations",
        "scope": "separate prospective follow-up; original primary gates unchanged",
        "science_hashes": {name: sha256_file(output / name) for name in SCIENCE_FILES},
    }
    write_json(output / "long_context_run_audit.json", audit)
    print(f"completed profile={profile} elapsed_seconds={perf_counter()-started:.1f}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    execution = commands.add_parser("run")
    execution.add_argument("--profile", choices=(SMOKE_PROFILE, PROFILE), required=True)
    execution.add_argument("--out", type=Path, required=True)
    evidence = commands.add_parser("test")
    evidence.add_argument("--junit", type=Path, required=True)
    evidence.add_argument("--out", type=Path, required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--smoke-a", type=Path, required=True)
    freezing.add_argument("--smoke-b", type=Path, required=True)
    freezing.add_argument("--test-evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.profile, args.out.resolve(), args.config.resolve())
    elif args.command == "test":
        run_bound_tests(args.junit.resolve(), args.out.resolve(), args.config)
    else:
        record = create_freeze(args.smoke_a, args.smoke_b, args.test_evidence, args.config, FREEZE)
        print(f"freeze_identity={record['material']['identity_sha256']}", flush=True)


if __name__ == "__main__":
    main()
