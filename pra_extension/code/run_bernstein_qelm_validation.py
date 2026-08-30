#!/usr/bin/env python3
"""Run and freeze the deterministic constructive Bernstein-QELM validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bernstein_qelm_validation import run_validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "pra_extension"
            / "outputs"
            / "bernstein_qelm_validation_v1"
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    material = list(rows)
    if not material:
        raise ValueError(f"cannot write empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(material[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(material)


def write_validation(output_dir: Path) -> dict[str, object]:
    """Run validation and write deterministic, content-addressed artifacts."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")

    result = run_validation()
    convergence_path = output_dir / "bernstein_convergence.csv"
    history_path = output_dir / "qrc_history_convergence.csv"
    summary_path = output_dir / "bernstein_validation_summary.json"
    manifest_path = output_dir / "artifact_sha256.json"

    convergence_rows = [
        *result["generic_convergence"],
        *result["qrc_bernstein_convergence"],
    ]
    _write_rows(convergence_path, convergence_rows)
    _write_rows(history_path, result["qrc_history_convergence"])
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.name: _sha256(path)
        for path in (convergence_path, history_path, summary_path)
    }
    manifest_path.write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "artifact_sha256": hashes,
        "feature_probability_audit": result["feature_probability_audit"],
        "contraction_audit": result["contraction_audit"],
        "generic_final": result["generic_convergence"][-1],
        "qrc_final": result["qrc_bernstein_convergence"][-1],
        "history_final": result["qrc_history_convergence"][-1],
    }


def main() -> int:
    args = parse_args()
    report = write_validation(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
