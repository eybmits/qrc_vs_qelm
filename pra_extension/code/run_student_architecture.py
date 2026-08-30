#!/usr/bin/env python3
"""Run the fresh-seed QELM student-architecture robustness arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from student_architecture import (
    default_architecture_config,
    run_architecture_study,
    summarize_architecture_results,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "student_architecture_results.csv"
    summary_path = output / "student_architecture_summary.json"
    if raw_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite an existing architecture output")

    started = time.perf_counter()
    config = default_architecture_config(args.profile)
    results = run_architecture_study(config)
    results.to_csv(raw_path, index=False)
    summary = summarize_architecture_results(results)
    summary["profile"] = args.profile
    summary["runtime_seconds"] = float(time.perf_counter() - started)
    summary["raw_results_sha256"] = _sha256(raw_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["counts"], sort_keys=True))
    print(f"runtime_seconds={summary['runtime_seconds']:.6f}")
    print(f"raw_results_sha256={summary['raw_results_sha256']}")


if __name__ == "__main__":
    main()
