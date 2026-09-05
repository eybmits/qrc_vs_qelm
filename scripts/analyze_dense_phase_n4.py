#!/usr/bin/env python
"""Validate and aggregate the frozen prospective four-qubit phase map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dense_phase_n4_common import (
    CONFIG_PATH,
    EXPECTED_BUDGETS,
    EXPECTED_GAMMAS,
    EXPECTED_ROOTS,
    EXPECTED_WINDOWS,
    FREEZE_PATH,
    PROFILE,
    ROOT,
    RUNNER_PATH,
    STUDY,
    assert_seed_disjointness,
    load_yaml,
    root_set_sha256,
    sha256_file,
    validate_dense_n4_configuration,
    verify_dense_n4_freeze,
)


VALID_REGIMES = {
    "DEGENERATE_TARGET",
    "HISTORY_IMPOSSIBLE",
    "EMULATED",
    "REPRESENTATION_LIMITED",
    "UNRESOLVED",
}
EXPECTED_ARTIFACTS = {
    "canonical_summary.json",
    "computational_resources.json",
    "stream_metrics.csv",
    "suffix_decomposition.csv",
    "system_diagnostics.csv",
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


def _expected_history_keys() -> set[tuple[Any, ...]]:
    return {
        (root, 4, gamma, window)
        for root in EXPECTED_ROOTS
        for gamma in EXPECTED_GAMMAS
        for window in EXPECTED_WINDOWS
    }


def _expected_budget_keys() -> set[tuple[Any, ...]]:
    return {
        (*key, budget)
        for key in _expected_history_keys()
        for budget in EXPECTED_BUDGETS
    }


def _validate_ten_models(frame: pd.DataFrame) -> None:
    keys = ["root_id", "teacher_n_qubits", "teacher_gamma", "window"]
    counts = frame.groupby(keys, sort=False).size()
    if len(counts) != 792 or not counts.eq(10).all():
        raise ValueError("each N4 root-grid cell must contain exactly ten model rows")
    expected_single = {"rr_same_identity", "constant", "linear_delay", "poly2_delay"}
    for _, cell in frame.groupby(keys, sort=False):
        model_counts = cell["model"].astype(str).value_counts().to_dict()
        if set(model_counts) != expected_single | {"ptd_parallel", "classical_tanh"}:
            raise ValueError("N4 root-grid cell contains an unexpected model family")
        if any(model_counts.get(model) != 1 for model in expected_single):
            raise ValueError("N4 root-grid cell lacks a required single-row control")
        if model_counts.get("ptd_parallel") != 3 or model_counts.get("classical_tanh") != 3:
            raise ValueError("N4 root-grid cell lacks the three frozen feature budgets")
        for model in ("ptd_parallel", "classical_tanh"):
            model_rows = cell[cell["model"].eq(model)].copy()
            snapshots = pd.to_numeric(model_rows["snapshot_budget"], errors="raise").astype(int)
            budgets = pd.to_numeric(model_rows["feature_budget"], errors="raise").astype(int)
            if set(zip(snapshots, budgets, strict=True)) != {(1, 18), (2, 36), (4, 72)}:
                raise ValueError(f"{model} snapshot-to-feature mapping has drifted")
    spread = frame.groupby(keys)["history_nrmse"].agg(
        lambda values: float(values.max() - values.min())
    )
    if float(spread.max()) > 1e-12:
        raise ValueError("history NRMSE changed across models or feature budgets")


def validate_results(results: Path, config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    results = Path(results).resolve()
    config_audit = validate_dense_n4_configuration(config_path)
    freeze_audit = verify_dense_n4_freeze(config_path)
    if not freeze_audit["passed"]:
        raise ValueError(f"N4 freeze audit failed: {freeze_audit['checks']}")
    freeze_record = _json(FREEZE_PATH)
    seed_audit = assert_seed_disjointness(
        config_path,
        freeze_record["prior_seed_source_files"].keys(),
    )
    if seed_audit["prior_seed_registry_sha256"] != freeze_audit["prior_seed_registry_sha256"]:
        raise ValueError("N4 prior-seed registry has drifted")

    manifest_path = results / "canonical_manifest.json"
    summary_path = results / "canonical_summary.json"
    stream_path = results / "stream_metrics.csv"
    suffix_path = results / "suffix_decomposition.csv"
    diagnostics_path = results / "system_diagnostics.csv"
    preflight_path = results / "dense_n4_preflight_audit.json"
    manifest = _json(manifest_path)
    summary = _json(summary_path)
    preflight = _json(preflight_path)
    if manifest.get("study") != STUDY or manifest.get("profile") != PROFILE:
        raise ValueError("run manifest has the wrong N4 study identity")
    if manifest.get("confirmatory") is not False:
        raise ValueError("N4 run must remain separate from the original confirmation")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("run manifest does not match the frozen N4 config")
    if manifest.get("runner_sha256") != sha256_file(RUNNER_PATH):
        raise ValueError("run manifest does not match the frozen N4 runner")
    canonical_module = ROOT / "src" / "qrc_vs_qelm" / "canonical_study.py"
    if manifest.get("module_sha256") != sha256_file(canonical_module):
        raise ValueError("run manifest does not match the frozen canonical module")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != EXPECTED_ARTIFACTS:
        raise ValueError("run manifest does not declare the exact canonical artifacts")
    for name in sorted(EXPECTED_ARTIFACTS):
        if artifacts.get(name) != sha256_file(results / name):
            raise ValueError(f"run artifact hash mismatch: {name}")
    if summary.get("study") != STUDY or summary.get("profile") != PROFILE:
        raise ValueError("run summary has the wrong N4 study identity")
    expected_counts = {
        "roots": 12,
        "systems": 132,
        "stream_rows": 7_920,
        "suffix_rows": 7_920,
        "target_live_systems": 132,
    }
    if summary.get("counts") != expected_counts:
        raise ValueError(f"unexpected N4 table counts: {summary.get('counts')}")
    if preflight.get("configuration") != config_audit:
        raise ValueError("N4 preflight configuration audit does not match")
    if preflight.get("freeze", {}).get("record_sha256") != sha256_file(FREEZE_PATH):
        raise ValueError("N4 preflight freeze record does not match")
    if preflight.get("freeze", {}).get("passed") is not True:
        raise ValueError("N4 preflight freeze audit did not pass")

    diagnostics = pd.read_csv(diagnostics_path)
    diagnostic_keys = ["root_id", "teacher_n_qubits", "teacher_gamma"]
    observed_systems = set(diagnostics[diagnostic_keys].itertuples(index=False, name=None))
    expected_systems = {
        (root, 4, gamma) for root in EXPECTED_ROOTS for gamma in EXPECTED_GAMMAS
    }
    if len(diagnostics) != 132 or diagnostics.duplicated(diagnostic_keys).any():
        raise ValueError("N4 diagnostics must contain 132 unique systems")
    if observed_systems != expected_systems:
        raise ValueError("N4 diagnostics do not form the frozen system grid")
    if not diagnostics["teacher_dissipator"].eq("local_loss").all():
        raise ValueError("N4 diagnostics contain a non-frozen dissipator")
    if not diagnostics["target_live"].astype(bool).all():
        raise ValueError("every N4 target system must be live")
    if not diagnostics["target_dim"].eq(30).all():
        raise ValueError("every N4 target system must expose 30 observables")
    target_variance = diagnostics["target_variance"].to_numpy(dtype=float)
    minimum_variance = float(load_yaml(config_path)["common"]["decision"]["min_target_variance"])
    if not np.isfinite(target_variance).all() or np.any(target_variance < minimum_variance):
        raise ValueError("N4 target variance is nonfinite or below the frozen live threshold")

    stream = pd.read_csv(stream_path)
    suffix = pd.read_csv(suffix_path)
    required = {
        "root_id",
        "teacher_n_qubits",
        "teacher_dissipator",
        "teacher_gamma",
        "window",
        "model",
        "snapshot_budget",
        "feature_budget",
        "history_nrmse",
        "history_mse",
        "representation_mse",
        "total_mse",
        "target_variance",
        "identity_error",
        "n_suffix_groups",
        "branches_per_group",
        "n_outputs",
        "regime",
    }
    if missing := required.difference(suffix.columns):
        raise ValueError(f"N4 suffix table lacks columns: {sorted(missing)}")
    if len(stream) != 7_920 or len(suffix) != 7_920:
        raise ValueError("N4 stream and suffix tables must each contain 7,920 rows")
    _validate_ten_models(stream.merge(
        suffix[[
            "root_id", "teacher_n_qubits", "teacher_gamma", "window", "model",
            "snapshot_budget", "feature_budget", "history_nrmse"
        ]],
        on=[
            "root_id", "teacher_n_qubits", "teacher_gamma", "window", "model",
            "snapshot_budget", "feature_budget"
        ],
        how="inner",
        validate="one_to_one",
    ))
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
        raise ValueError("N4 suffix metrics must be finite")
    if (suffix[["history_nrmse", "history_mse", "representation_mse", "total_mse"]].to_numpy() < 0).any():
        raise ValueError("N4 error metrics must be nonnegative")
    if float(np.max(np.abs(suffix["identity_error"].to_numpy(dtype=float)))) > 1e-10:
        raise ValueError("N4 decomposition identity exceeds 1e-10")
    if not suffix["n_suffix_groups"].eq(32).all():
        raise ValueError("N4 suffix-group count has drifted")
    if not suffix["branches_per_group"].eq(4).all():
        raise ValueError("N4 suffix branch count has drifted")
    if not suffix["n_outputs"].eq(30).all():
        raise ValueError("N4 suffix output count has drifted")
    if not suffix["teacher_dissipator"].eq("local_loss").all():
        raise ValueError("N4 suffix table contains a non-frozen dissipator")

    _validate_ten_models(suffix)
    history = suffix[suffix["model"].eq("rr_same_identity")].copy()
    history_key = ["root_id", "teacher_n_qubits", "teacher_gamma", "window"]
    observed_history = set(history[history_key].itertuples(index=False, name=None))
    if len(history) != 792 or history.duplicated(history_key).any():
        raise ValueError("N4 history rows are not one per root-grid cell")
    if observed_history != _expected_history_keys():
        raise ValueError("N4 history rows do not form the frozen Cartesian grid")

    ptd = suffix[suffix["model"].eq("ptd_parallel")].copy()
    ptd["feature_budget"] = pd.to_numeric(ptd["feature_budget"], errors="raise").astype(int)
    ptd_key = [*history_key, "feature_budget"]
    observed_ptd = set(ptd[ptd_key].itertuples(index=False, name=None))
    if len(ptd) != 2_376 or ptd.duplicated(ptd_key).any():
        raise ValueError("N4 QELM rows are not one per root-budget-grid cell")
    if observed_ptd != _expected_budget_keys():
        raise ValueError("N4 QELM rows do not form the frozen Cartesian grid")
    if not set(ptd["regime"].astype(str)).issubset(VALID_REGIMES):
        raise ValueError("N4 QELM rows contain an unknown regime")

    return {
        "stream": stream,
        "suffix": suffix,
        "history": history,
        "ptd": ptd,
        "diagnostics": diagnostics,
        "manifest": manifest,
        "summary": summary,
        "config_audit": config_audit,
        "seed_audit": seed_audit,
        "freeze_audit": freeze_audit,
        "run_manifest_sha256": sha256_file(manifest_path),
        "stream_sha256": sha256_file(stream_path),
        "suffix_sha256": sha256_file(suffix_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
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
    minimum_r_squared = float(config["common"]["decision"]["min_scaling_r_squared"])
    keys = ["teacher_n_qubits", "teacher_gamma", "window"]
    rows: list[dict[str, Any]] = []
    for key, cell_history in history.groupby(keys, sort=True):
        cell_ptd = ptd[
            ptd["teacher_n_qubits"].eq(key[0])
            & np.isclose(ptd["teacher_gamma"], key[1])
            & ptd["window"].eq(key[2])
        ]
        values = cell_history["history_nrmse"].to_numpy(dtype=float)
        history_sufficient_n = int(np.count_nonzero(values <= tolerance))
        row: dict[str, Any] = {
            "teacher_n_qubits": int(key[0]),
            "teacher_gamma": float(key[1]),
            "window": int(key[2]),
            "median_history_nrmse": float(np.median(values)),
            "minimum_history_nrmse": float(np.min(values)),
            "maximum_history_nrmse": float(np.max(values)),
            "history_n_roots": int(len(values)),
            "history_sufficient_n": history_sufficient_n,
        }
        successful: list[int] = []
        any_split = False
        agreements: list[int] = []
        modes: list[str] = []
        representation_counts: list[int] = []
        for budget in EXPECTED_BUDGETS:
            budget_cell = cell_ptd[cell_ptd["feature_budget"].eq(budget)]
            if len(budget_cell) != 12:
                raise ValueError("N4 budget cell must contain exactly 12 roots")
            modal, agreement, present = _modal_regime(budget_cell["regime"])
            emulated_n = int(budget_cell["regime"].eq("EMULATED").sum())
            representation_n = int(budget_cell["regime"].eq("REPRESENTATION_LIMITED").sum())
            row[f"modal_regime_B{budget}"] = modal
            row[f"agreement_n_B{budget}"] = agreement
            row[f"emulated_n_B{budget}"] = emulated_n
            row[f"representation_n_B{budget}"] = representation_n
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
        elif history_sufficient_n < minimum_successes:
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
                "evidence_cohort": "prospective_dense_n4_followup",
                "run_manifest_sha256": validated["run_manifest_sha256"],
                "root_set_sha256": validated["root_set_sha256"],
                "config_sha256": sha256_file(config_path),
                "stream_metrics_sha256": validated["stream_sha256"],
                "suffix_decomposition_sha256": validated["suffix_sha256"],
                "system_diagnostics_sha256": validated["diagnostics_sha256"],
                "freeze_record_sha256": validated["freeze_audit"]["record_sha256"],
            }
        )
        rows.append(row)
    cells = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    if len(cells) != 66 or cells.duplicated(keys).any():
        raise ValueError("N4 aggregate must contain 66 unique cells")

    scaling_rows: list[dict[str, Any]] = []
    positive = history[history["teacher_gamma"] > 0]
    for root, frame in positive.groupby("root_id", sort=True):
        x = (
            frame["teacher_gamma"].to_numpy(dtype=float)
            * float(config["common"]["teacher"]["dt"])
            * frame["window"].to_numpy(dtype=float)
        )
        y = np.log(np.maximum(frame["history_nrmse"].to_numpy(dtype=float), 1e-12))
        if len(frame) != 60 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("each N4 scaling fit requires 60 finite positive-loss cells")
        denominator = float(np.sum((y - np.mean(y)) ** 2))
        if denominator <= 0.0:
            raise ValueError("N4 scaling response has zero variance")
        slope, intercept = np.polyfit(x, y, deg=1)
        fitted = intercept + slope * x
        r_squared = 1.0 - float(np.sum((y - fitted) ** 2)) / denominator
        if not np.isfinite([slope, intercept, r_squared]).all():
            raise ValueError("N4 scaling fit is nonfinite")
        scaling_rows.append(
            {
                "root_id": str(root),
                "teacher_n_qubits": 4,
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_squared),
                "n_cells": int(len(frame)),
                "joint_direction_fit_passed": bool(slope < 0.0 and r_squared >= minimum_r_squared),
            }
        )
    scaling = pd.DataFrame(scaling_rows).sort_values("root_id").reset_index(drop=True)
    if len(scaling) != 12 or tuple(scaling["root_id"]) != EXPECTED_ROOTS:
        raise ValueError("N4 scaling table must contain the 12 frozen roots")
    joint_scaling_pass_n = int(scaling["joint_direction_fit_passed"].sum())
    fading_direction_pass = joint_scaling_pass_n >= minimum_successes

    frontier: dict[str, int] = {}
    for gamma in EXPECTED_GAMMAS:
        gamma_cells = cells[np.isclose(cells["teacher_gamma"], gamma)].sort_values("window")
        if len(gamma_cells) != 6:
            raise ValueError("N4 frontier requires six windows at every frozen loss rate")
        sufficient = gamma_cells[gamma_cells["history_sufficient_n"] >= minimum_successes]
        frontier[f"{gamma:g}"] = 7 if sufficient.empty else int(sufficient["window"].min())
    frontier_values = np.asarray(list(frontier.values()), dtype=int)
    differences = np.diff(frontier_values)
    frontier_nonincreasing = bool(np.all(differences <= 0))
    frontier_strict_step = bool(np.any(differences < 0))
    frontier_finite_count = int(np.count_nonzero(frontier_values <= 6))
    frontier_pass = bool(
        frontier_nonincreasing and frontier_strict_step and frontier_finite_count >= 4
    )

    anchor = ptd[
        np.isclose(ptd["teacher_gamma"], 4.0)
        & ptd["window"].eq(6)
        & ptd["feature_budget"].eq(72)
    ]
    if len(anchor) != 12:
        raise ValueError("N4 QELM anchor must contain exactly 12 roots")
    anchor_emulated_n = int(anchor["regime"].eq("EMULATED").sum())
    anchor_pass = anchor_emulated_n >= minimum_successes
    passed = bool(fading_direction_pass and frontier_pass and anchor_pass)
    summary = {
        "schema_version": 1,
        "study": STUDY,
        "profile": PROFILE,
        "decision": (
            "PASSES_DENSE_N4_EXTRAPOLATION"
            if passed
            else "DOES_NOT_PASS_DENSE_N4_EXTRAPOLATION"
        ),
        "passed": passed,
        "integrity_passed": True,
        "fading_direction_passed": fading_direction_pass,
        "joint_negative_slope_and_r_squared_roots": joint_scaling_pass_n,
        "measured_frontier_passed": frontier_pass,
        "frontier_d_star_censored_at_7": frontier,
        "frontier_nonincreasing": frontier_nonincreasing,
        "frontier_has_strict_step": frontier_strict_step,
        "frontier_finite_loss_rates": frontier_finite_count,
        "qelm_anchor_passed": anchor_pass,
        "anchor_emulated_roots_B72": anchor_emulated_n,
        "cell_state_counts": {
            str(key): int(value)
            for key, value in cells["state"].value_counts().sort_index().items()
        },
        "counts": {
            "aggregate_cells": int(len(cells)),
            "history_root_cells": int(len(history)),
            "qelm_root_budget_cells": int(len(ptd)),
            "scaling_root_fits": int(len(scaling)),
            "positive_loss_cells_per_fit": 60,
        },
        "minimum_median_history_nrmse": float(cells["median_history_nrmse"].min()),
        "maximum_median_history_nrmse": float(cells["median_history_nrmse"].max()),
        "run_manifest_sha256": validated["run_manifest_sha256"],
        "root_set_sha256": validated["root_set_sha256"],
        "config_sha256": sha256_file(config_path),
        "stream_metrics_sha256": validated["stream_sha256"],
        "suffix_decomposition_sha256": validated["suffix_sha256"],
        "system_diagnostics_sha256": validated["diagnostics_sha256"],
        "freeze_record_sha256": validated["freeze_audit"]["record_sha256"],
        "claim_boundary": (
            "teacher-size stability across tested two-, three-, and four-qubit "
            "local-loss ensembles under the fixed tested reset-QELM only"
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
    cells.to_csv(output / "dense_n4_phase_cells.csv", index=False, float_format="%.12g")
    scaling.to_csv(output / "dense_n4_root_scaling.csv", index=False, float_format="%.12g")
    (output / "dense_n4_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        name: sha256_file(output / name)
        for name in (
            "dense_n4_phase_cells.csv",
            "dense_n4_root_scaling.csv",
            "dense_n4_summary.json",
        )
    }
    manifest = {
        "schema_version": 1,
        "study": STUDY,
        "profile": PROFILE,
        "artifacts": artifacts,
        "config_sha256": sha256_file(args.config.resolve()),
        "freeze_record_sha256": validated["freeze_audit"]["record_sha256"],
        "run_manifest_sha256": validated["run_manifest_sha256"],
        "root_set_sha256": validated["root_set_sha256"],
    }
    (output / "dense_n4_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
