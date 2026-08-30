#!/usr/bin/env python3
"""Run the post-confirmatory all-splits-noisy finite-shot EIV study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from finite_shots_eiv import (  # noqa: E402
    default_eiv_config,
    eiv_seed_manifest,
    run_eiv_study,
    summarize_eiv_results,
)
from qrc_vs_qelm.provenance import (  # noqa: E402
    material_file_manifest,
    sha256_file,
    verify_canonical_freeze_record,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_audit() -> dict[str, object]:
    config = REPO_ROOT / "configs" / "canonical_emulation.yaml"
    record = REPO_ROOT / "configs" / "canonical_emulation_freeze.json"
    material = material_file_manifest(
        repo_root=REPO_ROOT,
        config_path=config,
        runner_path=REPO_ROOT / "scripts" / "run_canonical_emulation.py",
    )
    audit = verify_canonical_freeze_record(
        record,
        study="canonical_qrc_to_parallel_qelm_emulation",
        profile="confirmatory",
        config_sha256=sha256_file(config),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "pra_extension"
            / "outputs"
            / "finite_shots_eiv_n4_exploratory"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    before = freeze_audit()
    if not before["passed"]:
        raise RuntimeError("canonical freeze did not verify before EIV study")

    config = default_eiv_config(args.profile)
    ledger = eiv_seed_manifest(config)
    started = time.perf_counter()
    results = run_eiv_study(config)
    runtime = float(time.perf_counter() - started)
    results_path = output / "eiv_finite_shot_results.csv"
    results.to_csv(results_path, index=False, float_format="%.17g")

    after = freeze_audit()
    if not after["passed"] or (
        after["material_identity_sha256"] != before["material_identity_sha256"]
    ):
        raise RuntimeError("canonical material identity changed during EIV study")
    summary = summarize_eiv_results(results)
    summary.update(
        {
            "profile": args.profile,
            "runtime_seconds": runtime,
            "sampling_streams": {
                "entropy_tuples": int(len(ledger)),
                "unique_entropy_tuples": int(
                    len({row["entropy"] for row in ledger})
                ),
                "unique_seedsequence_states": int(
                    len({row["state_uint32"] for row in ledger})
                ),
                "split_streams_are_disjoint": True,
            },
            "canonical_freeze_before": before,
            "canonical_freeze_after": after,
            "artifact_sha256": {results_path.name: sha256(results_path)},
        }
    )
    summary_path = output / "eiv_finite_shot_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "rows": int(len(results)),
                "runtime_seconds": runtime,
                "sampling_streams": int(len(ledger)),
                "canonical_freeze_passed": after["passed"],
                "material_identity_sha256": after["material_identity_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
