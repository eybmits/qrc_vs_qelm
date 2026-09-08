"""Separately frozen, exact-feature learning curves; never alter old cohorts.

The reference is the exact conditional mean for a finite IID prefix of the
declared length, not a claim about stationary or arbitrary-history risk.
Smoke and benchmark profiles use development seeds and support engineering
checks only. Production requires a matching configuration and material freeze.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "src", ROOT / "scripts", Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from qrc_vs_qelm import canonical_study as canonical  # noqa: E402
from qrc_vs_qelm.canonical_emulation import (  # noqa: E402
    degree_two_delay_features, select_nested_snapshots,
)
from qrc_vs_qelm.emulation import fit_multivariate_ridge  # noqa: E402
from long_context_qelm_backend import efficient_student_channel  # noqa: E402

SNAPSHOTS = (1, 2, 4)
SEED_ROLES = (
    "teacher_coupling", "student_coupling", "train_input", "val_input",
    "test_input", "reference_windows", "panel_windows", "panel_prefixes",
)
REFERENCE_CONTRACT = (
    "exact conditional mean after a fixed-length IID finite-alphabet prefix "
    "from the all-zero state followed by the chronological reference window; "
    "not a stationary or uniform-history claim"
)


def config_file_sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _positive_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_config(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(), object_pairs_hook=_unique_json_pairs,
                     parse_constant=lambda value: (_ for _ in ()).throw(
                         ValueError(f"nonfinite JSON constant: {value}")))
    if not isinstance(raw, dict) or not isinstance(raw.get("study"), str) or not raw["study"]:
        raise ValueError("configuration requires a study name")
    rows = raw.get("training_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("training_rows must be nonempty")
    for value in rows:
        _positive_int("training_rows", value)
    if rows != sorted(set(rows)):
        raise ValueError("training_rows must be strictly increasing")
    for key in ("validation_rows", "test_rows", "reference_rows", "panel_groups", "prefix_length"):
        _positive_int(key, raw.get(key))
    _positive_int("prefix_branches", raw.get("prefix_branches"), 2)
    _positive_int("input_levels", raw.get("input_levels"), 2)
    if raw.get("max_snapshots") != 4:
        raise ValueError("max_snapshots must remain 4 with nested budgets 1, 2, 4")
    canonical._float_grid(raw.get("alpha_grid", []))
    anchors = raw.get("anchors")
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("anchors must be nonempty")
    lengths = raw.get("reference_prefix_lengths", [raw["prefix_length"], 2 * raw["prefix_length"]])
    if (not isinstance(lengths, list) or len(lengths) != 2
            or lengths[0] != raw["prefix_length"] or lengths[1] <= lengths[0]):
        raise ValueError("reference_prefix_lengths must start at prefix_length and increase")
    for length in lengths:
        _positive_int("reference_prefix_lengths", length)
    ids, codes = [], []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            raise ValueError("anchor must be a mapping")
        ident = anchor.get("id", "")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
            raise ValueError("anchor id must be a safe nonempty identifier")
        ids.append(ident)
        codes.append(_positive_int("anchor code", anchor.get("code", len(ids))))
        for key in ("teacher_n_qubits", "student_n_qubits", "window"):
            _positive_int(key, anchor.get(key))
        gamma = anchor.get("gamma")
        if isinstance(gamma, bool) or not isinstance(gamma, (float, int)) or not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        if anchor["window"] > min(anchor["student_n_qubits"], raw["prefix_length"]):
            raise ValueError("window must not exceed student register or washout")
    if len(ids) != len(set(ids)) or len(codes) != len(set(codes)):
        raise ValueError("anchor ids and codes must be unique")
    roots = raw.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("roots must be nonempty")
    ids, seeds = [], []
    for root in roots:
        ident = root.get("root_id", "")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
            raise ValueError("root_id must be a safe nonempty identifier")
        ids.append(ident)
        seeds.append(_positive_int("root seed", root.get("seed"), 0))
    seeds.append(_positive_int("development_seed", raw.get("development_seed"), 0))
    if "analysis_seed" in raw:
        seeds.append(_positive_int("analysis_seed", raw["analysis_seed"], 0))
    if len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
        raise ValueError("root ids and production/development seeds must be unique")
    return raw


def material_manifest(config_path: Path | None = None) -> dict[str, Any]:
    """Bind numerical source, task protocol, tests, analysis and config bytes."""
    paths = set((ROOT / "src" / "qrc_vs_qelm").rglob("*.py"))
    required = [
        ROOT / "pyproject.toml",
        ROOT / "pra_extension/TRAINING_DATA_CONTROL_PROTOCOL.md",
        ROOT / "pra_extension/code/run_training_data_control.py",
        ROOT / "pra_extension/code/conditional_mean_reference.py",
        ROOT / "pra_extension/code/analyze_training_data_control.py",
        ROOT / "pra_extension/code/freeze_training_data_control.py",
        ROOT / "pra_extension/tests/test_training_data_control.py",
        ROOT / "pra_extension/tests/test_conditional_mean_reference.py",
        ROOT / "scripts/long_context_qelm_backend.py",
        ROOT / "scripts/dense_phase_n4_common.py",
    ]
    if config_path is not None:
        required.append(Path(config_path).resolve())
    paths.update(required)
    paths.update((ROOT / "pra_extension/tests").glob("*training_data_control*.py"))
    files = {}
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"required frozen material missing: {path}")
        label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        files[label] = config_file_sha256(path)
    return {"files": files, "identity_sha256": sha256(_json_bytes(files)).hexdigest()}


def verify_freeze(config_path: Path, freeze_path: Path | None) -> dict[str, Any]:
    if freeze_path is None:
        raise ValueError("production requires --freeze")
    config = read_config(config_path)
    record = json.loads(Path(freeze_path).read_text(), object_pairs_hook=_unique_json_pairs)
    expected = {
        "status": "FROZEN_FOLLOWUP", "study": config["study"],
        "config_sha256": config_file_sha256(config_path),
        "material_sha256": material_manifest(config_path)["identity_sha256"],
        "root_ids": [root["root_id"] for root in config["roots"]],
        "production_roots_previously_opened": False,
    }
    for key, value in expected.items():
        if key not in record or record[key] != value or (value is False and record[key] is not False):
            raise ValueError(f"freeze mismatch: {key}")
    return record


def profile_config(config: dict[str, Any], profile: str) -> dict[str, Any]:
    cfg = deepcopy(config)
    if profile == "production":
        return cfg
    if profile not in {"smoke", "benchmark"}:
        raise ValueError(f"unknown profile: {profile}")
    cfg["roots"] = [{"root_id": f"development_{profile}", "seed": config["development_seed"]}]
    cfg.update(panel_groups=4, prefix_branches=4)
    if profile == "smoke":
        cfg.update(training_rows=[8, 16, 32, 64], validation_rows=24, test_rows=32,
                   reference_rows=16, prefix_length=8, reference_prefix_lengths=[8, 16])
        cfg["anchors"] = [{"id": "smoke", "teacher_n_qubits": 2,
                           "student_n_qubits": 3, "gamma": 1.0, "window": 2}]
    else:
        cfg.update(training_rows=[32, 64, 96, 128], validation_rows=48,
                   test_rows=64, reference_rows=32, prefix_length=128,
                   reference_prefix_lengths=[128, 256])
        cfg["anchors"] = [{"id": "benchmark_n5", "teacher_n_qubits": 5,
                           "student_n_qubits": 10, "gamma": 1.0, "window": 10}]
    return cfg


def seed_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate all derived seeds, including unopened production and dev roles."""
    ledger = []
    for profile_code, profile in enumerate(("production", "smoke", "benchmark")):
        cfg = profile_config(config, profile)
        for root in cfg["roots"]:
            for anchor_code, anchor in enumerate(cfg["anchors"], 1):
                for role_code, role in enumerate(SEED_ROLES, 1):
                    entropy = [root["seed"], profile_code, anchor.get("code", anchor_code), role_code]
                    seed = int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
                    ledger.append({"profile": profile, "root_id": root["root_id"],
                                   "anchor_id": anchor["id"], "role": role,
                                   "entropy": entropy, "seed": seed})
    if len({item["seed"] for item in ledger}) != len(ledger):
        raise ValueError("derived seed collision")
    if len({tuple(item["entropy"]) for item in ledger}) != len(ledger):
        raise ValueError("seed entropy collision")
    return ledger


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = _json_bytes({"shape": list(array.shape), "dtype": array.dtype.str})
    return sha256(header + b"\n" + array.tobytes()).hexdigest()


def _canonical_profile(cfg: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    return {
        "teacher": {"J_s": 1.0, "h": .5, "h_in": 1.0, "dt": .5,
                    "target_observables": {"axes": ["x", "y", "z"], "max_weight": 2}},
        "student": {"n_qubits": anchor["student_n_qubits"], "J_s": 1.0,
                    "h": .5, "T_evolve": .5,
                    "observables": {"axes": ["x", "y", "z"], "max_weight": 1}},
        "max_snapshots": cfg["max_snapshots"], "quantize_levels": cfg["input_levels"],
    }


def _draw(seed: int, shape: tuple[int, ...], levels: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, levels, size=shape).astype(float) / (levels - 1)


def _errors(name: str, predicted: np.ndarray, target: np.ndarray, variance: float) -> dict[str, float]:
    if predicted.shape != target.shape or not np.all(np.isfinite(predicted)):
        raise ValueError(f"invalid predictions for {name}")
    mse = float(np.mean((predicted - target) ** 2))
    return {f"{name}_mse": mse, f"{name}_nrmse": float(np.sqrt(mse / variance))}


def mean_state_integrity(state: np.ndarray) -> dict[str, Any]:
    """Audit without projecting, normalizing or changing the propagated state."""
    value = np.asarray(state)
    if (value.ndim != 2 or value.shape[0] != value.shape[1]
            or not value.size or not np.all(np.isfinite(value))):
        raise RuntimeError("mean prefix state must be a finite square matrix")
    trace = np.trace(value)
    trace_error = float(abs(trace - 1.0))
    hermiticity_error = float(np.max(np.abs(value - value.conj().T)))
    # Symmetrization is only for the eigenvalue diagnostic, never propagation.
    minimum_eigenvalue = float(np.linalg.eigvalsh((value + value.conj().T) / 2.0)[0])
    passed = (trace_error <= 1e-9 and hermiticity_error <= 1e-9
              and minimum_eigenvalue >= -1e-8)
    result = {"passed": bool(passed), "trace_real": float(trace.real),
              "trace_imaginary": float(trace.imag), "trace_error": trace_error,
              "hermiticity_max_abs_error": hermiticity_error,
              "minimum_eigenvalue": minimum_eigenvalue}
    if not passed:
        raise RuntimeError(f"mean prefix state integrity failed: {result}")
    return result


def fixed_test_target_variance(targets: np.ndarray) -> float:
    values = np.asarray(targets, dtype=float)
    if values.ndim != 2 or not values.size or not np.all(np.isfinite(values)):
        raise RuntimeError("test targets must be a nonempty finite matrix")
    variance = float(np.mean((values - values.mean(axis=0)) ** 2))
    if not np.isfinite(variance) or variance <= 1e-8:
        raise RuntimeError("test target variance must be strictly greater than 1e-8")
    return variance


def fit_curve_row(features: dict[str, np.ndarray], targets: dict[str, np.ndarray],
                  training_rows: int, alphas: tuple[float, ...]):
    """Only train+validation enter fitting; the scaler uses the nested train slice."""
    train = features["train"][:training_rows]
    train_y = targets["train"][:training_rows]
    values = np.concatenate([train, features["val"]])
    truth = np.concatenate([train_y, targets["val"]])
    model = fit_multivariate_ridge(values, truth, np.arange(len(train)),
                                  np.arange(len(train), len(values)), alphas=alphas)
    # Match the fitter's row-indexed C-order reduction, not the possibly
    # Fortran-ordered snapshot slice (whose roundoff can differ by one ULP).
    scaler_train = values[np.arange(len(train))]
    expected_scale = np.std(scaler_train, axis=0)
    expected_scale = np.where(expected_scale < 1e-12, 1.0, expected_scale)
    if not (np.array_equal(model.feature_mean, scaler_train.mean(axis=0))
            and np.array_equal(model.feature_scale, expected_scale)):
        raise RuntimeError("train-only standardization contract violated")
    return model


def run_anchor(cfg: dict[str, Any], profile: str, root: dict[str, Any],
               anchor: dict[str, Any], ledger: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    from conditional_mean_reference import mean_prefix_state, conditional_mean_outputs

    started = perf_counter()
    timings: dict[str, float] = {}
    selected = [item for item in ledger if item["profile"] == profile
                and item["root_id"] == root["root_id"] and item["anchor_id"] == anchor["id"]]
    seeds = {item["role"]: item["seed"] for item in selected}
    if set(seeds) != set(SEED_ROLES):
        raise ValueError("incomplete anchor seed ledger")
    canonical_profile = _canonical_profile(cfg, anchor)
    teacher = canonical._teacher_reservoir(
        canonical_profile, n_qubits=anchor["teacher_n_qubits"], dissipator="local_loss",
        gamma=anchor["gamma"], coupling_seed=seeds["teacher_coupling"])
    teacher_obs = canonical._teacher_observables(canonical_profile, anchor["teacher_n_qubits"])
    channel = efficient_student_channel(canonical_profile, seeds["student_coupling"])
    student_obs = canonical._student_observables(canonical_profile)
    timings["construct_seconds"] = perf_counter() - started
    stage = perf_counter()
    burnin, window, levels = cfg["prefix_length"], anchor["window"], cfg["input_levels"]
    inputs = {split: canonical._inputs(seeds[f"{split}_input"], burnin + count, levels)
              for split, count in (("train", max(cfg["training_rows"])),
                                   ("val", cfg["validation_rows"]), ("test", cfg["test_rows"]))}
    delays = canonical._delay_blocks(inputs, window=window, burnin=burnin)
    targets = canonical._teacher_blocks(teacher, inputs, teacher_obs, burnin=burnin)
    for split in inputs:
        if len(delays[split]) != len(targets[split]):
            raise RuntimeError("stream delay/target alignment mismatch")
    variance = fixed_test_target_variance(targets["test"])
    timings["stream_teacher_seconds"] = perf_counter() - stage
    stage = perf_counter()
    reference_windows = _draw(seeds["reference_windows"], (cfg["reference_rows"], window), levels)
    panel_windows = _draw(seeds["panel_windows"], (cfg["panel_groups"], window), levels)
    prefix_inputs = _draw(seeds["panel_prefixes"],
                          (cfg["panel_groups"], cfg["prefix_branches"], burnin), levels)
    mean_state = mean_prefix_state(teacher, np.linspace(0.0, 1.0, levels), burnin)
    mean_state_audit = mean_state_integrity(mean_state)
    reference_means = conditional_mean_outputs(teacher, mean_state, reference_windows, teacher_obs)
    long_prefix = cfg.get("reference_prefix_lengths", [burnin, 2 * burnin])[1]
    long_mean_state = mean_prefix_state(teacher, np.linspace(0.0, 1.0, levels), long_prefix)
    long_mean_state_audit = mean_state_integrity(long_mean_state)
    long_reference_means = conditional_mean_outputs(teacher, long_mean_state, reference_windows, teacher_obs)
    reference_difference = reference_means - long_reference_means
    prefix_stability = {
        "prefix_lengths": [burnin, long_prefix],
        "rms": float(np.sqrt(np.mean(reference_difference ** 2))),
        "max_abs": float(np.max(np.abs(reference_difference))),
        "normalized_rms": float(np.sqrt(np.mean(reference_difference ** 2) / variance)),
        "normalized_max_abs": float(np.max(np.abs(reference_difference)) / np.sqrt(variance)),
        "claim_boundary": "descriptive finite-prefix stability, not stationarity proof",
    }
    panel_exact_means = conditional_mean_outputs(teacher, mean_state, panel_windows, teacher_obs)
    prefix_states = canonical._prefix_states(teacher, prefix_inputs)
    panel_targets = canonical._suffix_targets(teacher, prefix_states, panel_windows, teacher_obs)
    panel_empirical_means = np.mean(panel_targets, axis=1)
    panel_history_mse = float(np.mean((panel_targets - panel_empirical_means[:, None, :]) ** 2))
    timings["conditional_reference_and_panel_seconds"] = perf_counter() - stage
    stage = perf_counter()
    # Teacher suffixes are chronological; canonical QELM and delay features are newest-first.
    windows = {**delays, "reference": reference_windows[:, ::-1], "panel": panel_windows[:, ::-1]}
    maximal_features = {name: channel.features(value, student_obs, encoding_scheme="one_to_one")
                        for name, value in windows.items()}
    poly_features = {name: degree_two_delay_features(value) for name, value in windows.items()}
    arrays = {**{f"input_{key}": value for key, value in inputs.items()},
              "reference_windows_chronological": reference_windows,
              "panel_windows_chronological": panel_windows, "panel_prefix_inputs": prefix_inputs,
              "mean_prefix_state": mean_state, "reference_conditional_means": reference_means,
              "long_mean_prefix_state": long_mean_state,
              "long_reference_conditional_means": long_reference_means,
              "panel_exact_means": panel_exact_means, "panel_targets": panel_targets,
              **{f"target_{key}": value for key, value in targets.items()},
              **{f"qelm_max_features_{key}": value for key, value in maximal_features.items()},
              **{f"poly2_features_{key}": value for key, value in poly_features.items()}}
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise RuntimeError("nonfinite generated inputs, targets, means or features")
    hashes = {key: array_sha256(value) for key, value in arrays.items()}
    timings["feature_generation_seconds"] = perf_counter() - stage
    stage = perf_counter()
    models = [("ptd_parallel", budget, {
        split: select_nested_snapshots(value, n_observables=len(student_obs),
                                       max_snapshots=4, selected_snapshots=budget)
        for split, value in maximal_features.items()}) for budget in SNAPSHOTS]
    models.append(("poly2_delay", 0, poly_features))
    rows = []
    alphas = canonical._float_grid(cfg["alpha_grid"])
    for count in cfg["training_rows"]:
        for name, snapshots, features in models:
            fitted = fit_curve_row(features, targets, count, alphas)
            singular, resolved_rank = canonical._resolved_singular_values(features["train"][:count])
            resolved_condition = (float(singular[0] / singular[resolved_rank - 1])
                                  if resolved_rank else None)
            predictions = {key: fitted.predict(value[:count] if key == "train" else value)
                           for key, value in features.items()}
            panel_total = float(np.mean((panel_targets - predictions["panel"][:, None, :]) ** 2))
            panel_residual = float(np.mean((predictions["panel"] - panel_empirical_means) ** 2))
            identity_error = abs(panel_total - panel_history_mse - panel_residual)
            if identity_error > 1e-10:
                raise RuntimeError("empirical panel decomposition failed")
            row = {
                "root_id": root["root_id"], "anchor_id": anchor["id"], "profile": profile,
                "teacher_n_qubits": anchor["teacher_n_qubits"],
                "student_n_qubits": anchor["student_n_qubits"], "gamma": anchor["gamma"],
                "window": window, "training_rows": count, "validation_rows": cfg["validation_rows"],
                "test_rows": cfg["test_rows"], "reference_rows": cfg["reference_rows"],
                "model": name, "snapshot_budget": snapshots, "feature_budget": features["train"].shape[1],
                "alpha": "inf" if np.isposinf(fitted.alpha) else float(fitted.alpha),
                "selection_validation_nrmse": fitted.validation_nrmse,
                "fit_rows": count + cfg["validation_rows"], "train_standardization_rows": count,
                "train_standardization_contract": "nested_training_rows_only_frozen_during_refit",
                "train_metrics_contract": "training_slice_scored_after_train_plus_validation_refit",
                "feature_mean_sha256": array_sha256(fitted.feature_mean),
                "feature_scale_sha256": array_sha256(fitted.feature_scale), "target_variance": variance,
                "resolved_feature_rank": resolved_rank, "resolved_condition_number": resolved_condition,
                "standardized_readout_weight_norm": float(np.linalg.norm(fitted.weights)),
                "reference_mean_sha256": hashes["reference_conditional_means"],
                "panel_history_mse": panel_history_mse, "panel_total_mse": panel_total,
                "panel_empirical_minus_exact_mean_mse": panel_residual - float(
                    np.mean((predictions["panel"] - panel_exact_means) ** 2)),
                "identity_error": identity_error,
                **_errors("train", predictions["train"], targets["train"][:count], variance),
                **_errors("test", predictions["test"], targets["test"], variance),
                **_errors("reference_mean", predictions["reference"], reference_means, variance),
                **_errors("panel_empirical_mean", predictions["panel"], panel_empirical_means, variance),
                **_errors("panel_exact_mean", predictions["panel"], panel_exact_means, variance),
                **_errors("panel_mean_sampling", panel_empirical_means, panel_exact_means, variance),
            }
            rows.append(row)
    timings["fit_and_score_seconds"] = perf_counter() - stage
    timings["total_seconds"] = perf_counter() - started
    manifest = {"anchor": anchor, "seeds": selected, "array_sha256": hashes,
                "train_slice_input_sha256": {str(count): array_sha256(inputs["train"][:burnin + count])
                                             for count in cfg["training_rows"]},
                "target_variance": variance, "reference_contract": REFERENCE_CONTRACT,
                "reference_prefix_length": burnin, "reference_rows": cfg["reference_rows"],
                "reference_prefix_stability": prefix_stability,
                "mean_prefix_state_integrity": {str(burnin): mean_state_audit,
                                                str(long_prefix): long_mean_state_audit},
                "panel_groups": cfg["panel_groups"], "prefix_branches": cfg["prefix_branches"],
                "reference_windows_order": "chronological", "student_windows_order": "newest_first",
                "panel_window_sampling": "IID with replacement; exact numerical-window mean retained",
                "timing_seconds": timings, "rows": len(rows),
                "training_rows": cfg["training_rows"], "max_identity_error": max(row["identity_error"] for row in rows)}
    return rows, manifest


def validate_coverage(table: pd.DataFrame, cfg: dict[str, Any], roots: list[dict]) -> dict[str, Any]:
    keys = ["root_id", "anchor_id", "training_rows", "model", "snapshot_budget"]
    expected = {(root["root_id"], anchor["id"], count, model, budget)
                for root in roots for anchor in cfg["anchors"] for count in cfg["training_rows"]
                for model, budget in [("ptd_parallel", 1), ("ptd_parallel", 2),
                                      ("ptd_parallel", 4), ("poly2_delay", 0)]}
    actual = set(table[keys].itertuples(index=False, name=None))
    if actual != expected or table.duplicated(keys).any():
        raise RuntimeError("result coverage differs from all registered settings")
    metrics = [name for name in table if name.endswith(("_mse", "_nrmse"))]
    if not np.isfinite(table[metrics].to_numpy(dtype=float)).all():
        raise RuntimeError("nonfinite result metric")
    if not (table["fit_rows"] == table["training_rows"] + cfg["validation_rows"]).all():
        raise RuntimeError("refit row count mismatch")
    for _, group in table.groupby(["root_id", "anchor_id"]):
        if group["target_variance"].nunique() != 1 or group["reference_mean_sha256"].nunique() != 1:
            raise RuntimeError("comparison does not use a fixed denominator/reference")
    return {"passed": True, "expected_rows": len(expected), "actual_rows": len(table)}


def run(config_path: Path, output: Path, *, profile: str = "smoke",
        root_id: str | None = None, freeze_path: Path | None = None) -> dict[str, Any]:
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = read_config(config_path)
    freeze = verify_freeze(config_path, freeze_path) if profile == "production" else None
    material = material_manifest(config_path)
    config_sha = config_file_sha256(config_path)
    cfg = profile_config(config, profile)
    ledger = seed_manifest(config)
    roots = cfg["roots"] if root_id is None else [item for item in cfg["roots"] if item["root_id"] == root_id]
    if not roots:
        raise ValueError("--root is not a root in the selected profile")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be absent or empty; never overwrite an existing run")
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    all_rows, root_records = [], []
    try:
        for root in roots:
            root_dir = output / root["root_id"]
            root_dir.mkdir()
            rows, anchors = [], []
            for anchor in cfg["anchors"]:
                print(f"[{profile}] {root['root_id']} / {anchor['id']}", flush=True)
                anchor_rows, anchor_record = run_anchor(cfg, profile, root, anchor, ledger)
                rows.extend(anchor_rows)
                anchors.append(anchor_record)
            table = pd.DataFrame(rows)
            coverage = validate_coverage(table, cfg, [root])
            table.to_csv(root_dir / "metrics.csv", index=False, float_format="%.17g")
            record = {"schema_version": 1, "study": config["study"], "profile": profile,
                      "root_id": root["root_id"], "config_sha256": config_sha,
                      "material_sha256": material["identity_sha256"],
                      "anchors": anchors, "coverage": coverage,
                      "metrics_sha256": config_file_sha256(root_dir / "metrics.csv")}
            (root_dir / "manifest.json").write_bytes(_json_bytes(record) + b"\n")
            all_rows.extend(rows)
            root_records.append({"root_id": root["root_id"],
                                 "manifest_sha256": config_file_sha256(root_dir / "manifest.json")})
        table = pd.DataFrame(all_rows)
        coverage = validate_coverage(table, cfg, roots)
        if config_file_sha256(config_path) != config_sha or material_manifest(config_path) != material:
            raise RuntimeError("configuration or frozen material changed during execution")
        table.to_csv(output / "metrics.csv", index=False, float_format="%.17g")
        summary = {"schema_version": 1, "study": config["study"], "profile": profile,
                   "status": "COMPLETED", "scientific_evidence": profile == "production",
                   "claim_boundary": "learning curves only; no quantum advantage or uniform approximation claim",
                   "reference_contract": REFERENCE_CONTRACT, "config_sha256": config_file_sha256(config_path),
                   "material_sha256": material["identity_sha256"], "material_manifest": material,
                   "effective_config": cfg, "selected_root_ids": [root["root_id"] for root in roots],
                   "production_freeze": freeze, "seed_manifest": ledger, "coverage": coverage,
                   "roots": root_records, "metrics_sha256": config_file_sha256(output / "metrics.csv"),
                   "runtime_seconds": perf_counter() - started}
        (output / "manifest.json").write_bytes(_json_bytes(summary) + b"\n")
        return summary
    except Exception as error:
        (output / "failure.json").write_bytes(_json_bytes({
            "status": "FAILED", "profile": profile, "config_sha256": config_file_sha256(config_path),
            "error_type": type(error).__name__, "error": str(error),
            "completed_roots": root_records, "runtime_seconds": perf_counter() - started}) + b"\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "benchmark", "production"), default="smoke")
    parser.add_argument("--root", dest="root_id")
    parser.add_argument("--freeze", type=Path, dest="freeze_path")
    args = parser.parse_args()
    result = run(args.config, args.output, profile=args.profile,
                 root_id=args.root_id, freeze_path=args.freeze_path)
    print(json.dumps({"status": result["status"], "coverage": result["coverage"]}))


if __name__ == "__main__":
    main()
