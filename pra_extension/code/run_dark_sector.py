#!/usr/bin/env python3
"""Run the frozen protected--excited--visible dark-sector validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dark_sector import (  # noqa: E402
    addressable_x_pulse,
    asymmetric_singlet_initial_decay,
    collective_jumps,
    evolve_pair,
    expectation,
    singlet_audit,
    singlet_projector,
    symmetric_two_qubit_terms,
    zero_state,
)
from qrc_vs_qelm.core import dissipators, liouvillian  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "dark_sector_validation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "pra_extension" / "outputs" / "dark_sector_validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")
    config = _load_config(config_path)
    architecture = config["architecture"]
    theta = float(architecture["pulse_angle"])
    dt = float(architecture["dt"])
    gamma = float(architecture["gamma"])
    field = float(architecture["field"])
    drive = float(architecture["drive"])
    steps = int(architecture["suffix_steps"])
    first = zero_state()
    second = addressable_x_pulse(first, theta, site=0)
    expected_gap = float(np.sin(theta / 2.0) ** 2 / 2.0)

    trajectory_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    symmetric_unexcited_max = 0.0
    for root in config["roots"]:
        root_id = str(root["root_id"])
        coupling_rng = np.random.default_rng(int(root["coupling_seed"]))
        suffix_rng = np.random.default_rng(int(root["suffix_seed"]))
        coupling = float(coupling_rng.uniform(-1.0, 1.0))
        suffix = suffix_rng.uniform(0.0, 1.0, size=steps)
        h_static, h_drive = symmetric_two_qubit_terms(coupling, field, drive)

        protected_jumps = collective_jumps(gamma)
        for value in (0.0, 0.5, 1.0):
            audit = singlet_audit(h_static + value * h_drive, protected_jumps)
            audit_rows.append({
                "root_id": root_id,
                "coupling": coupling,
                "suffix_input": value,
                "family": "collective_loss",
                "asymmetry": 0.0,
                **audit.to_dict(),
            })

        unexcited = zero_state()
        projector = singlet_projector()
        for value in suffix:
            generator = liouvillian.lindbladian(h_static + value * h_drive, protected_jumps)
            unexcited = liouvillian.evolve_step(generator, unexcited, dt)
            symmetric_unexcited_max = max(
                symmetric_unexcited_max, abs(expectation(unexcited, projector))
            )

        families: list[tuple[str, float, list[tuple[np.ndarray, float]]]] = [
            ("collective_loss", 0.0, protected_jumps),
            ("local_loss", 0.0, dissipators.local_loss(2, gamma)),
        ]
        families.extend(
            (
                "asymmetric_collective_loss",
                float(epsilon),
                collective_jumps(gamma, float(epsilon)),
            )
            for epsilon in config["perturbations"]["collective_jump_asymmetries"]
        )
        for family, epsilon, jumps in families:
            rows = evolve_pair(first, second, h_static, h_drive, jumps, suffix, dt)
            for row in rows:
                trajectory_rows.append({
                    "root_id": root_id,
                    "coupling": coupling,
                    "family": family,
                    "asymmetry": epsilon,
                    **row,
                })

        for epsilon in config["perturbations"]["collective_jump_asymmetries"]:
            leakage_rows.append({
                "root_id": root_id,
                "asymmetry": float(epsilon),
                "predicted_initial_decay": asymmetric_singlet_initial_decay(
                    gamma, float(epsilon)
                ),
            })

    trajectories = pd.DataFrame(trajectory_rows)
    audits = pd.DataFrame(audit_rows)
    leakage = pd.DataFrame(leakage_rows)
    protected = trajectories[trajectories["family"].eq("collective_loss")]
    initial = protected[protected["step"].eq(0)].set_index("root_id")
    merged = protected.join(
        initial[["population_gap", "suffix_only_mse_floor"]],
        on="root_id",
        rsuffix="_initial",
    )
    checks = {
        "collective_adjoint_invariance_max": float(
            audits["adjoint_invariance_error"].max()
        ),
        "collective_population_drift_max": float(
            (merged["population_gap"] - merged["population_gap_initial"]).abs().max()
        ),
        "analytic_prefix_gap_error_max": float(
            (initial["population_gap"] - expected_gap).abs().max()
        ),
        "common_suffix_floor_error_max": float(
            (
                initial["suffix_only_mse_floor"]
                - initial["population_gap"] ** 2 / 4.0
            ).abs().max()
        ),
        "symmetric_unexcited_population_max": float(symmetric_unexcited_max),
    }
    thresholds = {
        key: float(value) for key, value in config["frozen_primary_checks"].items()
    }
    passed = {key: bool(checks[key] <= threshold) for key, threshold in thresholds.items()}
    if not all(passed.values()):
        raise RuntimeError(f"frozen dark-sector check failed: {passed}")

    trajectory_path = output_dir / "dark_sector_trajectories.csv"
    audit_path = output_dir / "dark_sector_audits.csv"
    leakage_path = output_dir / "dark_sector_leakage.csv"
    summary_path = output_dir / "dark_sector_summary.json"
    trajectories.to_csv(trajectory_path, index=False)
    audits.to_csv(audit_path, index=False)
    leakage.to_csv(leakage_path, index=False)
    endpoint = trajectories[trajectories["step"].eq(steps)]
    summary = {
        "study": config["study"],
        "claim_scope": config["claim_scope"],
        "config_sha256": _sha256(config_path),
        "n_roots": int(len(config["roots"])),
        "suffix_steps": steps,
        "expected_prefix_gap": expected_gap,
        "expected_balanced_mse_floor": float(expected_gap ** 2 / 4.0),
        "expected_worst_case_absolute_floor": float(expected_gap / 2.0),
        "primary_checks": checks,
        "primary_thresholds": thresholds,
        "primary_passed": passed,
        "all_primary_passed": bool(all(passed.values())),
        "endpoint_median_absolute_gap": {
            str(family): float(group["population_gap"].abs().median())
            for family, group in endpoint.groupby("family")
        },
        "maximum_singlet_projector_negativity": 0.5,
    }
    summary["artifact_sha256"] = {
        trajectory_path.name: _sha256(trajectory_path),
        audit_path.name: _sha256(audit_path),
        leakage_path.name: _sha256(leakage_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
