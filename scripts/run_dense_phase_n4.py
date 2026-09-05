#!/usr/bin/env python
"""Run the separately frozen prospective four-qubit dense phase map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrc_vs_qelm.canonical_study import run_canonical_study

from dense_phase_n4_common import (
    FREEZE_PATH,
    assert_seed_disjointness,
    validate_dense_n4_configuration,
    verify_dense_n4_freeze,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "canonical_emulation_dense_n4.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = args.config.resolve()
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    configuration_audit = validate_dense_n4_configuration(config)
    freeze_audit = verify_dense_n4_freeze(config)
    if not freeze_audit["passed"]:
        raise SystemExit(f"N4 freeze verification failed: {freeze_audit['checks']}")
    seed_audit = assert_seed_disjointness(
        config,
        freeze_audit_source_files(),
    )
    if (
        seed_audit["prior_seed_registry_sha256"]
        != freeze_audit["prior_seed_registry_sha256"]
    ):
        raise SystemExit("N4 prior-seed registry no longer matches the freeze record")
    summary = run_canonical_study(
        config_path=config,
        profile_name="dense_n4",
        output_dir=output,
        runner_path=Path(__file__).resolve(),
    )
    audit = {
        "schema_version": 1,
        "configuration": configuration_audit,
        "seed_disjointness": seed_audit,
        "freeze": freeze_audit,
    }
    (output / "dense_n4_preflight_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary)
    return 0


def freeze_audit_source_files() -> tuple[str, ...]:
    """Return exactly the prior seed sources committed into the freeze record."""
    record = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    sources = record.get("prior_seed_source_files")
    if not isinstance(sources, dict):
        raise ValueError("N4 freeze record lacks a prior-seed source registry")
    return tuple(sorted(str(path) for path in sources))


if __name__ == "__main__":
    raise SystemExit(main())
