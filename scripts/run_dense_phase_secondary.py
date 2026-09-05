#!/usr/bin/env python
"""Run the prospectively specified dense local-loss secondary replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrc_vs_qelm.canonical_study import run_canonical_study

from dense_phase_secondary_common import (
    assert_seed_disjointness,
    validate_dense_configuration,
    verify_dense_freeze,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "canonical_emulation_dense_secondary.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    configuration_audit = validate_dense_configuration(args.config.resolve())
    seed_audit = assert_seed_disjointness(args.config.resolve())
    freeze_audit = verify_dense_freeze(args.config.resolve())
    if not freeze_audit["passed"]:
        raise SystemExit(f"dense freeze verification failed: {freeze_audit['checks']}")
    summary = run_canonical_study(
        config_path=args.config.resolve(),
        profile_name="dense_secondary",
        output_dir=output,
        runner_path=Path(__file__).resolve(),
    )
    audit = {
        "schema_version": 1,
        "configuration": configuration_audit,
        "seed_disjointness": seed_audit,
        "freeze": freeze_audit,
    }
    (output / "dense_secondary_preflight_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
