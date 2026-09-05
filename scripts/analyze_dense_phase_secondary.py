#!/usr/bin/env python
"""Validate and aggregate the frozen dense phase-map secondary replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dense_phase_secondary_common import (
    CONFIG_PATH,
    EXPECTED_BUDGETS,
    EXPECTED_GAMMAS,
    EXPECTED_ROOTS,
    EXPECTED_WINDOWS,
    PROFILE,
    STUDY,
    assert_seed_disjointness,
    load_yaml,
    root_set_sha256,
    sha256_file,
    validate_dense_configuration,
    verify_dense_freeze,
)


VALID_REGIMES = {
    "DEGENERATE_TARGET",
    "HISTORY_IMPOSSIBLE",
    "EMULATED",
    "REPRESENTATION_LIMITED",
    "UNRESOLVED",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _modal_regime(values: pd.Series) -> tuple[str, int, str]:
    counts = values.astype(str).value_counts().to_dict()
    unknown = set(counts).difference(VALID_REGIMES)
    if unknown:
        raise ValueError(f"unknown root-level regimes: {sorted(unknown)}")
    best = max(counts.values())
    winners = sorted(name for name, count in counts.items() if count == best)
    modal = winners[0] if len(winners) == 1 else "MIXED"
    present = ";".join(f"{name}:{counts[name]}" for name in sorted(counts))
    return modal, int(best), present


def _expected_keys(with_budget: bool) -> set[tuple[Any, ...]]:
    keys = {
        (root, teacher, gamma, window)
        for root in EXPECTED_ROOTS
        for teacher in (2, 3)
        for gamma in EXPECTED_GAMMAS
        for window in EXPECTED_WINDOWS
    }
    if not with_budget:
        return keys
    return {(*key, budget) for key in keys for budget in EXPECTED_BUDGETS}


def validate_results(results: Path, config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    results = Path(results).resolve()
    config_audit = validate_dense_configuration(config_path)
    seed_audit = assert_seed_disjointness(config_path)
    freeze_audit = verify_dense_freeze(config_path)
    if not freeze_audit["passed"]:
        raise ValueError(f"dense freeze audit failed: {freeze_audit['checks']}")

    manifest_path = results / "canonical_manifest.json"
    summary_path = results / "canonical_summary.json"
    suffix_path = results / "suffix_decomposition.csv"
    manifest = _json(manifest_path)
    summary = _json(summary_path)
    if manifest.get("study") != STUDY or manifest.get("profile") != PROFILE:
        raise ValueError("run manifest has the wrong dense study identity")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("run manifest does not match the frozen dense config")
    for name in ("canonical_summary.json", "suffix_decomposition.csv"):
        if manifest.get("artifacts", {}).get(name) != sha256_file(results / name):
            raise ValueError(f"run artifact hash mismatch: {name}")
    if summary.get("study") != STUDY or summary.get("profile") != PROFILE:
        raise ValueError("run summary has the wrong dense study identity")
    expected_counts = {
        "roots": 12,
        "systems": 264,
        "stream_rows": 15_840,
        "suffix_rows": 15_840,
        "target_live_systems": 264,
    }
    if summary.get("counts") != expected_counts:
        raise ValueError(f"unexpected dense table counts: {summary.get('counts')}")

    suffix = pd.read_csv(suffix_path)
    required = {
        "root_id",
        "teacher_n_qubits",
        "teacher_gamma",
        "window",
        "model",
        "feature_budget",
        "history_nrmse",
        "history_mse",
        "representation_mse",
        "total_mse",
        "target_variance",
        "identity_error",
        "regime",
    }
    if missing := required.difference(suffix.columns):
        raise ValueError(f"dense suffix table lacks columns: {sorted(missing)}")
    if len(suffix) != 15_840:
        raise ValueError("dense suffix table must contain 15,840 rows")
    numeric = suffix[
        [
            "history_nrmse",
            "history_mse",
            "representation_mse",
            "total_mse",
            "target_variance",
            "identity_error",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("dense suffix metrics must be finite")
    if float(np.max(np.abs(suffix["identity_error"].to_numpy(dtype=float)))) > 1e-10:
        raise ValueError("dense decomposition identity exceeds 1e-10")

    history = suffix[suffix["model"].eq("rr_same_identity")].copy()
    history_key = ["root_id", "teacher_n_qubits", "teacher_gamma", "window"]
    observed_history = set(history[history_key].itertuples(index=False, name=None))
    if len(history) != 1_584 or history.duplicated(history_key).any():
        raise ValueError("dense history rows are not one per root-grid cell")
    if observed_history != _expected_keys(with_budget=False):
        raise ValueError("dense history rows do not form the frozen Cartesian grid")
    if (history["history_nrmse"] <= 0.0).any():
        raise ValueError("dense history NRMSE values must be positive")

    ptd = suffix[suffix["model"].eq("ptd_parallel")].copy()
    ptd["feature_budget"] = pd.to_numeric(
        ptd["feature_budget"], errors="raise"
    ).astype(int)
    ptd_key = [*history_key, "feature_budget"]
    observed_ptd = set(ptd[ptd_key].itertuples(index=False, name=None))
    if len(ptd) != 4_752 or ptd.duplicated(ptd_key).any():
        raise ValueError("dense QELM rows are not one per root-budget-grid cell")
    if observed_ptd != _expected_keys(with_budget=True):
        raise ValueError("dense QELM rows do not form the frozen Cartesian grid")
    spread = ptd.groupby(history_key)["history_nrmse"].agg(
        lambda values: float(values.max() - values.min())
    )
    if float(spread.max()) > 1e-12:
        raise ValueError("history NRMSE changed across feature budgets")
    if not set(ptd["regime"].astype(str)).issubset(VALID_REGIMES):
        raise ValueError("dense QELM rows contain an unknown regime")

    return {
        "suffix": suffix,
        "history": history,
        "ptd": ptd,
        "manifest": manifest,
        "summary": summary,
        "config_audit": config_audit,
        "seed_audit": seed_audit,
        "freeze_audit": freeze_audit,
        "run_manifest_sha256": sha256_file(manifest_path),
        "suffix_sha256": sha256_file(suffix_path),
        "root_set_sha256": root_set_sha256(config_path),
    }


def aggregate(
    validated: dict[str, Any],
    config_path: Path = CONFIG_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    history = validated["history"]
    ptd = validated["ptd"]
    config = load_yaml(config_path)
    tolerance = float(config["common"]["decision"]["tolerance_nrmse"])
    minimum_successes = int(config["common"]["decision"]["min_root_successes"])
    keys = ["teacher_n_qubits", "teacher_gamma", "window"]
    rows: list[dict[str, Any]] = []
    for key, cell_history in history.groupby(keys, sort=True):
        cell_ptd = ptd[
            ptd["teacher_n_qubits"].eq(key[0])
            & np.isclose(ptd["teacher_gamma"], key[1])
            & ptd["window"].eq(key[2])
        ]
        values = cell_history["history_nrmse"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "teacher_n_qubits": int(key[0]),
            "teacher_gamma": float(key[1]),
            "window": int(key[2]),
            "median_history_nrmse": float(np.median(values)),
            "minimum_history_nrmse": float(np.min(values)),
            "maximum_history_nrmse": float(np.max(values)),
            "history_n_roots": int(len(values)),
        }
        successful: list[int] = []
        any_split = False
        agreements: list[int] = []
        modes: list[str] = []
        representation_counts: list[int] = []
        for budget in EXPECTED_BUDGETS:
            budget_cell = cell_ptd[cell_ptd["feature_budget"].eq(budget)]
            modal, agreement, present = _modal_regime(budget_cell["regime"])
            emulated_n = int(budget_cell["regime"].eq("EMULATED").sum())
            representation_n = int(
                budget_cell["regime"].eq("REPRESENTATION_LIMITED").sum()
            )
            row[f"modal_regime_B{budget}"] = modal
            row[f"agreement_n_B{budget}"] = agreement
            row[f"emulated_n_B{budget}"] = emulated_n
            row[f"regimes_present_B{budget}"] = present
            if emulated_n >= minimum_successes:
                successful.append(budget)
            any_split = any_split or agreement < 12
            agreements.append(agreement)
            modes.append(modal)
            representation_counts.append(representation_n)
        if successful:
            state = "FEASIBLE"
            minimum_budget: float = float(min(successful))
        elif row["median_history_nrmse"] > tolerance:
            state = "HISTORY_MISSING"
            minimum_budget = np.nan
        elif max(representation_counts) >= minimum_successes:
            state = "MAP_NOT_REPRESENTED"
            minimum_budget = np.nan
        else:
            state = "UNRESOLVED"
            minimum_budget = np.nan
        row.update(
            {
                "state": state,
                "minimum_feature_budget": minimum_budget,
                "any_root_disagreement": any_split,
                "modal_regimes_B18_B36_B72": ";".join(modes),
                "minimum_agreement_n": min(agreements),
                "run_manifest_sha256": validated["run_manifest_sha256"],
                "root_set_sha256": validated["root_set_sha256"],
                "config_sha256": sha256_file(config_path),
                "suffix_decomposition_sha256": validated["suffix_sha256"],
                "freeze_record_sha256": validated["freeze_audit"]["record_sha256"],
            }
        )
        rows.append(row)
    cells = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    if len(cells) != 132 or cells.duplicated(keys).any():
        raise ValueError("dense aggregate must contain 132 unique cells")

    scaling_rows: list[dict[str, Any]] = []
    positive = history[history["teacher_gamma"] > 0]
    for (root, teacher), frame in positive.groupby(
        ["root_id", "teacher_n_qubits"], sort=True
    ):
        x = (
            frame["teacher_gamma"].to_numpy(dtype=float)
            * float(config["common"]["teacher"]["dt"])
            * frame["window"].to_numpy(dtype=float)
        )
        y = np.log(
            np.maximum(frame["history_nrmse"].to_numpy(dtype=float), 1e-12)
        )
        slope, intercept = np.polyfit(x, y, deg=1)
        fitted = intercept + slope * x
        denominator = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - float(np.sum((y - fitted) ** 2)) / denominator
        scaling_rows.append(
            {
                "root_id": str(root),
                "teacher_n_qubits": int(teacher),
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_squared),
                "n_cells": int(len(frame)),
            }
        )
    scaling = pd.DataFrame(scaling_rows).sort_values(
        ["teacher_n_qubits", "root_id"]
    ).reset_index(drop=True)
    negative_by_teacher = {
        str(teacher): int(group["slope"].lt(0.0).sum())
        for teacher, group in scaling.groupby("teacher_n_qubits", sort=True)
    }
    threshold_coverage = {
        str(teacher): {
            "above": int(group["median_history_nrmse"].gt(tolerance).sum()),
            "at_or_below": int(group["median_history_nrmse"].le(tolerance).sum()),
        }
        for teacher, group in cells.groupby("teacher_n_qubits", sort=True)
    }
    anchor = cells[
        np.isclose(cells["teacher_gamma"], 4.0) & cells["window"].eq(6)
    ]
    anchor_emulated = {
        str(int(row.teacher_n_qubits)): int(row.emulated_n_B72)
        for row in anchor.itertuples(index=False)
    }
    directional_pass = (
        set(negative_by_teacher) == {"2", "3"}
        and all(value >= minimum_successes for value in negative_by_teacher.values())
    )
    coverage_pass = all(
        value["above"] > 0 and value["at_or_below"] > 0
        for value in threshold_coverage.values()
    )
    anchor_pass = set(anchor_emulated) == {"2", "3"} and all(
        value >= minimum_successes for value in anchor_emulated.values()
    )
    passed = bool(directional_pass and coverage_pass and anchor_pass)
    summary = {
        "schema_version": 1,
        "study": STUDY,
        "profile": PROFILE,
        "decision": (
            "PASSES_DENSE_SECONDARY"
            if passed
            else "DOES_NOT_PASS_DENSE_SECONDARY"
        ),
        "passed": passed,
        "directional_prediction_passed": directional_pass,
        "threshold_coverage_passed": coverage_pass,
        "qelm_anchor_passed": anchor_pass,
        "negative_slopes_by_teacher": negative_by_teacher,
        "threshold_coverage_cells_by_teacher": threshold_coverage,
        "anchor_emulated_roots_B72_by_teacher": anchor_emulated,
        "cell_state_counts": {
            str(key): int(value)
            for key, value in cells["state"].value_counts().sort_index().items()
        },
        "counts": {
            "aggregate_cells": int(len(cells)),
            "history_root_cells": int(len(history)),
            "qelm_root_budget_cells": int(len(ptd)),
            "scaling_root_fits": int(len(scaling)),
        },
        "minimum_median_history_nrmse": float(
            cells["median_history_nrmse"].min()
        ),
        "maximum_median_history_nrmse": float(
            cells["median_history_nrmse"].max()
        ),
        "run_manifest_sha256": validated["run_manifest_sha256"],
        "root_set_sha256": validated["root_set_sha256"],
        "config_sha256": sha256_file(config_path),
        "suffix_decomposition_sha256": validated["suffix_sha256"],
        "freeze_record_sha256": validated["freeze_audit"]["record_sha256"],
        "claim_boundary": (
            "tested local-loss QRC families and tested six-qubit reset-QELM only"
        ),
    }
    return cells, scaling, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty analysis directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    validated = validate_results(args.results, args.config.resolve())
    cells, scaling, summary = aggregate(validated, args.config.resolve())
    cells.to_csv(
        output / "dense_phase_cells.csv",
        index=False,
        float_format="%.12g",
    )
    scaling.to_csv(
        output / "dense_phase_root_scaling.csv",
        index=False,
        float_format="%.12g",
    )
    (output / "dense_phase_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        name: sha256_file(output / name)
        for name in (
            "dense_phase_cells.csv",
            "dense_phase_root_scaling.csv",
            "dense_phase_summary.json",
        )
    }
    (output / "dense_phase_analysis_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "artifacts": artifacts},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
