#!/usr/bin/env python
"""Validate and summarize every cell of the separate prospective N5 follow-up."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qrc_vs_qelm.canonical_emulation import classify_emulation_regime
from qrc_vs_qelm.canonical_resources import observable_names_sha256
from qrc_vs_qelm.core import readout

from long_context_n5_common import (
    BUDGETS, CONFIG, GAMMAS, PROFILE, ROOT, ROOTS, SCIENCE_FILES, STUDY, WINDOWS,
    configuration, material_manifest, median_bootstrap, read_json, sha256_file,
    verify_freeze, write_json,
)

KEYS = ["root_id", "teacher_n_qubits", "teacher_gamma", "window"]
EXPECTED_MODELS = {"rr_same_identity": 1, "constant": 1, "linear_delay": 1,
                   "poly2_delay": 1, "ptd_parallel": 3, "classical_tanh": 3}
BACKEND = "identical_dense_unitaries_direct_product_encoding_sparse_local_pauli_expectations"


def _validate_target_metadata(frame: pd.DataFrame, profile: dict) -> None:
    required = {"root_id", "teacher_n_qubits", "teacher_gamma", "system_id",
                "teacher_dissipator", "teacher_coupling_seed", "target_observables_sha256"}
    if not required.issubset(frame.columns):
        raise ValueError(f"N5 missing target metadata: {sorted(required - set(frame.columns))}")
    seeds = {root["root_id"]: root["teacher_coupling_seeds"]["5"] for root in profile["roots"]}
    target_hash = observable_names_sha256(
        item.name for item in readout.pauli_observables(5, max_weight=2)
    )
    if not frame["target_observables_sha256"].eq(target_hash).all():
        raise ValueError("N5 declared target observable hash drift")
    if not frame["teacher_dissipator"].eq("local_loss").all():
        raise ValueError("N5 target dissipator drift")
    for row in frame.itertuples(index=False):
        if row.root_id not in seeds or row.teacher_n_qubits != 5 or row.teacher_gamma not in GAMMAS:
            raise ValueError("N5 target metadata is outside the fixed system grid")
        gamma_label = f"{float(row.teacher_gamma):.3f}".replace(".", "p")
        if row.system_id != f"{row.root_id}_n5_local_loss_g{gamma_label}":
            raise ValueError("N5 system identifier does not match its root/target/loss")
        if row.teacher_coupling_seed != seeds[row.root_id]:
            raise ValueError("N5 target coupling seed does not match its configured root")


def validate_tables(stream: pd.DataFrame, suffix: pd.DataFrame, systems: pd.DataFrame) -> dict:
    profile = configuration()
    expected_keys = {(root, 5, gamma, window) for root in ROOTS for gamma in GAMMAS for window in WINDOWS}
    if len(stream) != 1200 or len(suffix) != 1200 or len(systems) != 24:
        raise ValueError("N5 complete-table coverage failed")
    system_keys = ["root_id", "teacher_n_qubits", "teacher_gamma"]
    if systems.duplicated(system_keys).any() or set(map(tuple, systems[system_keys].to_numpy())) != {
        (root, 5, gamma) for root in ROOTS for gamma in GAMMAS
    }:
        raise ValueError("N5 target-system grid failed")
    for frame in (stream, suffix, systems):
        _validate_target_metadata(frame, profile)
    if not systems["target_live"].eq(True).all() or not np.isfinite(systems["target_variance"]).all() or not systems["target_variance"].ge(1e-8).all():
        raise ValueError("N5 target liveness failed; roots may not be discarded")
    if not systems["target_dim"].eq(45).all():
        raise ValueError("N5 must retain all 45 declared target outputs")
    student_hash = observable_names_sha256(f"{axis}{site}" for axis in "xyz" for site in range(10))
    for frame in (stream, suffix):
        observed = set(map(tuple, frame[KEYS].drop_duplicates().to_numpy()))
        if observed != expected_keys:
            raise ValueError("N5 root-loss-window coverage failed")
        if not frame["student_n_qubits"].eq(10).all() or not frame["student_observables_sha256"].eq(student_hash).all():
            raise ValueError("N5 full-window feature contract failed")
        if not frame["teacher_dissipator"].eq("local_loss").all():
            raise ValueError("N5 target dissipator drift")
        for _, cell in frame.groupby(KEYS):
            if cell["model"].value_counts().to_dict() != EXPECTED_MODELS:
                raise ValueError("N5 cell must retain every control and all budgets")
            for model in ("ptd_parallel", "classical_tanh"):
                selected = cell.loc[cell["model"].eq(model)]
                if set(zip(selected["snapshot_budget"], selected["feature_budget"])) != {(1, 30), (2, 60), (4, 120)}:
                    raise ValueError("N5 nested budget contract failed")
            window = int(cell["window"].iloc[0])
            for model, width in (("rr_same_identity", 45), ("constant", 0),
                                 ("linear_delay", window),
                                 ("poly2_delay", window + window * (window + 1) // 2)):
                selected = cell.loc[cell["model"].eq(model)]
                if not selected["feature_budget"].eq(width).all() or not selected["snapshot_budget"].isna().all():
                    raise ValueError("N5 control feature/snapshot contract failed")
    quantities = suffix[["history_mse", "representation_mse", "total_mse", "target_variance",
                         "history_nrmse", "representation_nrmse", "total_nrmse", "identity_error"]].to_numpy(float)
    if not np.isfinite(quantities).all() or (quantities[:, :3] < 0).any() or (quantities[:, 3] < 1e-8).any():
        raise ValueError("N5 suffix quantities are nonfinite, negative or degenerate")
    identity = np.abs(suffix["total_mse"] - suffix["history_mse"] - suffix["representation_mse"])
    if identity.max() > 1e-10 or suffix["identity_error"].abs().max() > 1e-10:
        raise ValueError("N5 exact suffix decomposition failed")
    for prefix in ("history", "representation", "total"):
        expected = np.sqrt(suffix[f"{prefix}_mse"] / suffix["target_variance"])
        if not np.allclose(suffix[f"{prefix}_nrmse"], expected, rtol=1e-10, atol=1e-12):
            raise ValueError(f"N5 stored {prefix} NRMSE disagrees with MSE")
    for column in ("predictions_agree", "rms_half_bound_satisfied", "diameter_half_bound_satisfied", "suffix_bounds_satisfied"):
        if not suffix[column].eq(True).all():
            raise ValueError(f"N5 suffix audit failed: {column}")
    diagnostic_columns = ["prediction_disagreement", "repeated_suffix_rmse", "worst_branch_rmse",
                          "rms_pair_half_bound", "max_group_diameter", "half_max_group_diameter",
                          "rms_bound_slack", "diameter_bound_slack"]
    if not set(diagnostic_columns).issubset(suffix.columns):
        raise ValueError("N5 missing numeric common-suffix audit quantities")
    diagnostics = suffix[diagnostic_columns].to_numpy(float)
    if not np.isfinite(diagnostics).all() or (diagnostics[:, :6] < 0).any():
        raise ValueError("N5 common-suffix numeric audit is nonfinite or negative")
    if suffix["prediction_disagreement"].gt(1e-12).any():
        raise ValueError("N5 identical-window predictions disagree numerically")
    if (diagnostics[:, 6:] < -1e-12).any():
        raise ValueError("N5 numeric common-suffix lower bound failed")
    expected_diagnostics = {
        "half_max_group_diameter": suffix["max_group_diameter"] / 2,
        "rms_bound_slack": suffix["repeated_suffix_rmse"] - suffix["rms_pair_half_bound"],
        "diameter_bound_slack": suffix["worst_branch_rmse"] - suffix["max_group_diameter"] / 2,
        "repeated_suffix_rmse": np.sqrt(suffix["total_mse"]),
    }
    for column, expected in expected_diagnostics.items():
        if not np.allclose(suffix[column], expected, rtol=1e-10, atol=2e-12):
            raise ValueError(f"N5 stored {column} disagrees with numeric suffix quantities")
    if (suffix["worst_branch_rmse"] + 1e-12 < suffix["repeated_suffix_rmse"]).any():
        raise ValueError("N5 worst-branch error is smaller than aggregate suffix error")
    replay = suffix.loc[suffix["model"].eq("rr_same_identity")].set_index(KEYS)["total_nrmse"]
    for _, cell in suffix.groupby(KEYS):
        for column in ("history_mse", "target_variance", "history_nrmse"):
            if cell[column].max() - cell[column].min() > 1e-12:
                raise ValueError("N5 model-dependent history quantity")
    for row in suffix.itertuples(index=False):
        key = (row.root_id, row.teacher_n_qubits, row.teacher_gamma, row.window)
        regime = classify_emulation_regime(
            history_nrmse=row.history_nrmse, student_nrmse=row.total_nrmse,
            oracle_nrmse=float(replay.loc[key]), target_variance=row.target_variance,
            tolerance=0.10, min_target_variance=1e-8,
        )
        if regime != row.regime:
            raise ValueError("N5 stored regime disagrees with recomputed quantities")
    stream_numeric = stream[["mse", "rmse", "nrmse"]].to_numpy(float)
    if not np.isfinite(stream_numeric).all() or (stream_numeric < 0).any():
        raise ValueError("N5 stream errors are invalid")
    if not np.allclose(stream["rmse"], np.sqrt(stream["mse"]), rtol=1e-10, atol=1e-12):
        raise ValueError("N5 stream RMSE disagrees with MSE")
    variances = systems.set_index("system_id")["target_variance"]
    expected_nrmse = np.sqrt(stream["mse"].to_numpy() / stream["system_id"].map(variances).to_numpy())
    if not np.allclose(stream["nrmse"], expected_nrmse, rtol=1e-10, atol=1e-12):
        raise ValueError("N5 stream NRMSE disagrees with target variance")
    return {"passed": True, "systems": 24, "live_systems": 24, "root_cells": 120,
            "qelm_root_budget_cells": 360, "stream_rows": 1200, "suffix_rows": 1200,
            "aggregate_loss_window_cells": 10, "max_identity_error": float(identity.max()),
            "max_prediction_disagreement": float(suffix["prediction_disagreement"].max())}


def summarize_cells(suffix: pd.DataFrame) -> pd.DataFrame:
    history = suffix.loc[suffix["model"].eq("rr_same_identity")]
    qelm = suffix.loc[suffix["model"].eq("ptd_parallel")]
    rows = []
    for (gamma, window), group in history.groupby(["teacher_gamma", "window"]):
        values = group["history_nrmse"]
        count = int(values.le(0.10).sum())
        row = {"teacher_n_qubits": 5, "student_n_qubits": 10, "teacher_gamma": float(gamma),
               "window": int(window), "n_roots": 12, "evidence_cohort": "prospective_long_context_n5",
               "median_history_nrmse": float(values.median()), "minimum_history_nrmse": float(values.min()),
               "maximum_history_nrmse": float(values.max()), "q10_history_nrmse": float(values.quantile(.1)),
               "q90_history_nrmse": float(values.quantile(.9)), "history_sufficient_n": count}
        successful = []
        for budget in BUDGETS:
            cell = qelm.loc[qelm["teacher_gamma"].eq(gamma) & qelm["window"].eq(window) & qelm["feature_budget"].eq(budget)]
            if len(cell) != 12:
                raise ValueError("budget summary must contain exactly twelve roots")
            n = int(cell["regime"].eq("EMULATED").sum())
            row[f"emulated_n_B{budget}"] = n
            for metric in ("total_nrmse", "representation_nrmse"):
                row[f"median_{metric}_B{budget}"] = float(cell[metric].median())
                row[f"q10_{metric}_B{budget}"] = float(cell[metric].quantile(.1))
                row[f"q90_{metric}_B{budget}"] = float(cell[metric].quantile(.9))
            if n >= 10:
                successful.append(budget)
        minimum = min(successful) if successful else None
        if minimum is not None and count < 10:
            raise ValueError("robust QELM success without robust history")
        row["minimum_feature_budget"] = minimum
        row["display_outcome"] = (
            "HISTORY_NOT_ROBUST" if count < 10 else f"QELM_{minimum}" if minimum is not None else "NO_TESTED_QELM_ROBUST"
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["teacher_gamma", "window"]).reset_index(drop=True)


def paired_contrasts(suffix: pd.DataFrame, stream: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    history = suffix.loc[suffix["model"].eq("rr_same_identity")].set_index(["root_id", "teacher_gamma", "window"])
    rows, summaries = [], {}
    specifications = [(1.0, 2, 10), (4.0, 2, 10), (1.0, 6, 10)]
    for index, (gamma, first, last) in enumerate(specifications):
        values = []
        label = f"history_gamma{gamma:g}_d{first}_to_d{last}"
        for root in ROOTS:
            before = float(history.loc[(root, gamma, first), "history_nrmse"])
            after = float(history.loc[(root, gamma, last), "history_nrmse"])
            if before <= 0:
                raise ValueError("relative history contrast has a zero denominator")
            gain = 1.0 - after / before
            values.append(gain)
            rows.append({"contrast": label, "root_id": root, "teacher_gamma": gamma,
                         "from_window": first, "to_window": last, "before": before, "after": after,
                         "relative_reduction": gain, "newly_history_sufficient": before > .10 and after <= .10})
        summary = median_bootstrap(values, 96_990_001 + index)
        if index < 2:
            summary["directional_prediction_passed"] = summary["positive_roots"] >= 10
        else:
            summary["interpretation"] = "descriptive longer-than-original-window contrast"
        summary["newly_history_sufficient_roots"] = sum(
            int(row["newly_history_sufficient"]) for row in rows if row["contrast"] == label
        )
        summaries[label] = summary
    comparison_rows, comparison_summaries = [], {}
    for cell_index, (gamma, window) in enumerate((g, w) for g in GAMMAS for w in WINDOWS):
        values = []
        for root in ROOTS:
            cell = stream.loc[stream["root_id"].eq(root) & stream["teacher_gamma"].eq(gamma) & stream["window"].eq(window)]
            quantum = float(cell.loc[cell["model"].eq("ptd_parallel") & cell["feature_budget"].eq(120), "nrmse"].iloc[0])
            quadratic = float(cell.loc[cell["model"].eq("poly2_delay"), "nrmse"].iloc[0])
            if quadratic <= 0:
                raise ValueError("classical relative contrast has a zero denominator")
            gain = 1.0 - quantum / quadratic
            values.append(gain)
            comparison_rows.append({"root_id": root, "teacher_gamma": gamma, "window": window,
                                    "qelm_B120_test_nrmse": quantum, "poly2_test_nrmse": quadratic,
                                    "qelm_relative_gain_vs_poly2": gain})
        comparison_summaries[f"gamma{gamma:g}_d{window}"] = median_bootstrap(values, 96_990_101 + cell_index)
    return pd.DataFrame(rows), pd.DataFrame(comparison_rows), {
        "paired_history": summaries, "fixed_B120_vs_quadratic_stream": comparison_summaries,
        "comparison_scope": "fixed-budget descriptive controls; no test-selected best-budget policy",
    }


def validate_run_records(results: Path, audit: dict, manifest: dict, freeze: dict) -> None:
    """Require the complete canonical artifact set and the frozen code identity."""
    if audit.get("schema_version") != 1 or audit.get("backend") != BACKEND:
        raise ValueError("N5 run audit schema or numerical backend drift")
    if audit.get("profile") != PROFILE or audit.get("material") != material_manifest(CONFIG) or audit.get("freeze") != freeze:
        raise ValueError("run audit does not match the frozen N5 material")
    if manifest.get("schema_version") != 1 or manifest.get("study") != STUDY or manifest.get("profile") != PROFILE or manifest.get("confirmatory") is not False:
        raise ValueError("canonical run identity does not match the follow-up")
    if manifest.get("config_path") != str(CONFIG.resolve()):
        raise ValueError("canonical run configuration path does not match the follow-up")
    expected_hashes = {
        "config_sha256": sha256_file(CONFIG),
        "module_sha256": sha256_file(ROOT / "src/qrc_vs_qelm/canonical_study.py"),
        "runner_sha256": sha256_file(ROOT / "scripts/run_long_context_n5.py"),
    }
    if any(manifest.get(name) != digest for name, digest in expected_hashes.items()):
        raise ValueError("canonical config/module/runner hashes do not match frozen material")
    artifacts = manifest.get("artifacts")
    science = audit.get("science_hashes")
    if not isinstance(artifacts, dict) or set(artifacts) != {*SCIENCE_FILES, "computational_resources.json"}:
        raise ValueError("canonical manifest lacks the exact complete artifact set")
    if not isinstance(science, dict) or set(science) != set(SCIENCE_FILES):
        raise ValueError("N5 run audit lacks the exact complete science artifact set")
    for name, digest in artifacts.items():
        if sha256_file(results / name) != digest:
            raise ValueError(f"run artifact hash mismatch: {name}")
    for name, digest in science.items():
        if artifacts[name] != digest:
            raise ValueError(f"science hash mismatch: {name}")


def analyze(results: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite nonempty N5 analysis output")
    freeze = verify_freeze()
    audit = read_json(results / "long_context_run_audit.json")
    manifest = read_json(results / "canonical_manifest.json")
    validate_run_records(results, audit, manifest, freeze)
    stream = pd.read_csv(results / "stream_metrics.csv")
    suffix = pd.read_csv(results / "suffix_decomposition.csv")
    systems = pd.read_csv(results / "system_diagnostics.csv")
    integrity = validate_tables(stream, suffix, systems)
    cells = summarize_cells(suffix)
    contrasts, comparisons, summaries = paired_contrasts(suffix, stream)
    control_source = stream[[*KEYS, "model", "feature_budget", "nrmse", "validation_nrmse"]].copy()
    summary = {"schema_version": 1, "study": STUDY, "profile": PROFILE, "integrity": integrity,
               "freeze": freeze, "outcome_counts": cells["display_outcome"].value_counts().to_dict(),
               "analysis": summaries, "old_gates_changed": False,
               "claim_boundary": "prospective five-qubit/ten-input local-loss extension; no universal or quantum-advantage claim"}
    output.mkdir(parents=True, exist_ok=False)
    cells.to_csv(output / "long_n5_cells.csv", index=False)
    contrasts.to_csv(output / "long_n5_history_root_contrasts.csv", index=False)
    comparisons.to_csv(output / "long_n5_classical_root_comparisons.csv", index=False)
    control_source.to_csv(output / "long_n5_stream_plot_source.csv", index=False)
    write_json(output / "long_n5_summary.json", summary)
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "long_n5_analysis_manifest.json", {
        "schema_version": 1, "material_identity_sha256": audit["material"]["identity_sha256"],
        "run_audit_sha256": sha256_file(results / "long_context_run_audit.json"),
        "upstream_canonical_manifest_sha256": sha256_file(results / "canonical_manifest.json"),
        "artifacts": artifacts,
    })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.results.resolve(), args.out.resolve())
    print(result)
