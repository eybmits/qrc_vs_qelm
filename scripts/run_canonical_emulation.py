#!/usr/bin/env python
"""Run the sealed canonical QRC-to-parallel-QELM emulation study."""

from __future__ import annotations

import argparse
from pathlib import Path

from qrc_vs_qelm.canonical_study import run_canonical_study


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "calibration", "confirmatory"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "canonical_emulation.yaml",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {out}")
    summary = run_canonical_study(
        config_path=args.config.resolve(),
        profile_name=args.profile,
        output_dir=out,
        runner_path=Path(__file__).resolve(),
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
