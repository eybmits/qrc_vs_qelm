"""Independent checkpoint audit for the every-window NARMA extension.

Inputs, targets, scores, saved linear predictions and ridge fitting are checked
without calling the extension's evaluator, feature generator or scoring helper.
Full original-batch features audit the six archived windows on every root;
direct chronological reservoir evolution audits fixed test rows at every d.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from laser_demonstration import ALPHAS, ROOT, architecture
from laser_forecast import replay_features
import task_memory_replication as replication

PARENT = ROOT / "paper/figures/task_memory/fig04_task_memory_source.json"
DEFAULT_RESULTS = ROOT / "results_narma_dense_replay_v2"
WINDOWS = tuple(range(1, 49))
ANCHORS = (1, 4, 8, 12, 24, 48)
# The explicit v2 addendum concerns comparison with the archived runtime only.
ARCHIVE_ATOL = 1e-5
FEATURE_ATOL = 1e-10
PREDICTION_ATOL = 1e-10
PREDICTION_RTOL = 1e-9
FIT_SCORE_ATOL = 1e-8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True)
    return hashlib.sha256(header.encode() + b"\n" + value.tobytes()).hexdigest()


def independent_data(seed, cfg):
    """Regenerate separate raw blocks and literal NARMA recurrence locally."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7301]))
    raw, blocks, times, targets = [], [], [], []
    for size in cfg.block_lengths:
        u = rng.uniform(size=size)
        y = np.zeros(size)
        for t in range(cfg.order - 1, size - 1):
            feedback = sum(float(y[t - lag]) for lag in range(cfg.order))
            y[t + 1] = (.3 * y[t] + .05 * y[t] * feedback
                        + 1.5 * cfg.input_scale**2 * u[t - cfg.order + 1] * u[t] + .1)
        index = np.arange(cfg.washout, size - 1)
        raw.append(u)
        blocks.append(np.rint(u * (cfg.levels - 1)) / (cfg.levels - 1))
        times.append(index)
        targets.append(y[index + 1])
    return raw, blocks, times, targets


def saved_prediction(saved, key, features):
    x = (features - saved[key + "_mean"]) / saved[key + "_scale"]
    return ((x @ saved[key + "_weights"] + saved[key + "_intercept"])
            * saved[key + "_target_sd"] + saved[key + "_target_mean"])


def check_score(row, prediction, target, training_sd):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if (prediction.shape != target.shape or not np.isfinite(prediction).all()
            or not np.isfinite(training_sd) or training_sd <= 0):
        raise ValueError("Invalid prediction or target scale")
    rmse = float(np.linalg.norm(prediction - target) / np.sqrt(target.size))
    expected = np.array([rmse, rmse / training_sd])
    actual = np.array([row["rmse"], row["nrmse"]])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-13)
    return float(np.max(np.abs(actual - expected)))


def check_archive(row, old):
    for key in ("model", "window", "gamma", "features", "alpha"):
        if row[key] != old[key]:
            raise ValueError(f"Archived {key} changed")
    differences = {key: abs(float(row[key]) - float(old[key]))
                   for key in ("validation_mse", "rmse", "nrmse")}
    np.testing.assert_allclose([row[k] for k in differences], [old[k] for k in differences],
                               rtol=0, atol=ARCHIVE_ATOL)
    return differences


def independent_candidates(x_train, y_train, x_validation, y_validation):
    """Literal frozen ridge equations and validation selection, locally coded."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    xt, xv = (x_train - mean) / scale, (x_validation - mean) / scale
    center, target_center = xt.mean(axis=0), float(y_train.mean())
    xc, yc = xt - center, y_train - target_center
    gram, rhs = xc.T @ xc, xc.T @ yc
    candidates = []
    for alpha in ALPHAS:
        weights = np.linalg.solve(gram + alpha * np.eye(xc.shape[1]), rhs)
        intercept = target_center - float(center @ weights)
        error = float(np.mean((xv @ weights + intercept - y_validation)**2))
        candidates.append({"alpha": alpha, "validation_mse": error})
    return mean, scale, gram, rhs, center, target_center, candidates


def check_fit(saved, key, row, features, targets):
    """Check all training moments and the original validation-only alpha grid."""
    target_mean, target_sd = float(targets[0].mean()), float(targets[0].std())
    yt, yv = (targets[0] - target_mean) / target_sd, (targets[1] - target_mean) / target_sd
    mean, scale, gram, rhs, center, target_center, candidates = independent_candidates(
        features[0], yt, features[1], yv)
    np.testing.assert_allclose(saved[key + "_mean"], mean, rtol=0, atol=FEATURE_ATOL)
    np.testing.assert_allclose(saved[key + "_scale"], scale, rtol=0, atol=FEATURE_ATOL)
    selected = min(candidates, key=lambda item: item["validation_mse"])
    if float(saved[key + "_alpha"]) != row["alpha"] or selected["alpha"] != row["alpha"]:
        raise ValueError("Readout alpha is not the training-fit validation minimum")
    prediction = saved_prediction(saved, key, features[1])
    error = float(np.mean(((prediction - targets[1]) / target_sd)**2))
    np.testing.assert_allclose(error, row["validation_mse"], rtol=0, atol=FIT_SCORE_ATOL)
    weights = saved[key + "_weights"]
    matrix = gram + row["alpha"] * np.eye(gram.shape[0])
    residual = float(np.linalg.norm(matrix @ weights - rhs)
                     / max(np.linalg.norm(matrix) * np.linalg.norm(weights) + np.linalg.norm(rhs), 1.))
    if residual > 1e-10:
        raise ValueError("Saved weights do not solve the training-only ridge equations")
    np.testing.assert_allclose(saved[key + "_intercept"], target_center - center @ weights,
                               rtol=0, atol=1e-10)
    return {"validation_score_difference": abs(error - row["validation_mse"]),
            "ridge_backward_error": residual,
            "ridge_condition_number": float(np.linalg.cond(matrix)),
            "candidate_validation_scores": candidates}


def check_prediction(saved, key, features, picks=None):
    expected = saved[key] if picks is None else saved[key][picks]
    predicted = saved_prediction(saved, key, features)
    np.testing.assert_allclose(predicted, expected, rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL)
    return float(np.max(np.abs(predicted - expected)))


def check_provenance(results):
    cfg = replication.configs()["narma"]
    specification = results / "extension_specification.json"
    spec = json.loads(specification.read_text())
    parent = json.loads(PARENT.read_text())
    config = json.loads(json.dumps(asdict(cfg)))
    if spec["config"] != config or parent["study_summaries"]["narma"]["config"] != config:
        raise ValueError("NARMA configuration changed")
    if (spec["roots"] != list(cfg.seeds) or spec["windows"] != list(WINDOWS)
            or spec["anchor_windows"] != list(ANCHORS) or spec["alphas"] != list(ALPHAS)):
        raise ValueError("Cohort, window coverage or tuning grid changed")
    if spec["parent_sha256"] != digest(PARENT):
        raise ValueError("Archived parent identity changed")
    if spec["metric_tolerance"] != {"rtol": 0., "atol": ARCHIVE_ATOL}:
        raise ValueError("Expected the explicit v2 archive-only numerical tolerance")
    for relative, expected in spec["sources"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen dependency: {relative}")
    records, identities = [], {"extension_specification.json": digest(specification)}
    for seed in cfg.seeds:
        path = results / f"seed{seed}.json"
        record = json.loads(path.read_text())
        if record["seed"] != seed or record["specification_sha256"] != digest(specification):
            raise ValueError("Root or specification identity mismatch")
        array_path = results / record["array_file"]
        if array_path.name != f"seed{seed}.npz" or digest(array_path) != record["array_sha256"]:
            raise ValueError("Checkpoint array identity mismatch")
        if [r["window"] for r in record["rows"]] != list(WINDOWS):
            raise ValueError("Missing, duplicated or reordered replay windows")
        for row in record["rows"]:
            if (row["seed"] != seed or row["model"] != "replay" or row["features"] != 48
                    or row["alpha"] not in ALPHAS
                    or row["gamma"] != spec["selected_qrc_gamma"][str(seed)]):
                raise ValueError("Changed replay architecture or selected loss")
        identities[path.name], identities[array_path.name] = digest(path), digest(array_path)
        records.append(record)
    return cfg, parent, spec, records, identities


def verify(results=DEFAULT_RESULTS, output=None):
    results = Path(results)
    cfg, parent, spec, records, identities = check_provenance(results)
    cells = {c["seed"]: c for c in parent["study_summaries"]["narma"]["results"]}
    maxima = {k: 0. for k in ("score_recomputation", "prediction_reconstruction", "direct_vs_original_features",
                              "archive_validation_mse", "archive_rmse", "archive_nrmse", "archive_trace",
                              "validation_score_recomputation", "ridge_backward_error", "ridge_condition_number")}
    counts = {k: 0 for k in ("roots", "scores", "archived_replay_anchors", "archived_continuous_qrc",
                             "direct_test_predictions", "full_anchor_predictions", "full_readout_audits")}
    root_reports = []
    for record in records:
        seed, cell = record["seed"], cells[record["seed"]]
        raw, blocks, times, targets = independent_data(seed, cfg)
        if record["gamma"] != cell["selection"]["qrc"]:
            raise ValueError("Frozen QRC loss changed")
        if record["scored_rows"] != [len(t) for t in times]:
            raise ValueError("Split rows changed")
        with np.load(results / record["array_file"], allow_pickle=False) as saved:
            for label, u, target in zip(("train", "validation", "test"), raw, targets, strict=True):
                np.testing.assert_array_equal(saved[label + "_raw_input"], u)
                np.testing.assert_allclose(saved[label + "_target"], target, rtol=1e-14, atol=1e-14)
                if record["input_sha256"][label] != array_digest(u):
                    raise ValueError("Raw-input digest differs")
            np.testing.assert_array_equal(saved["test_indices"], times[2])
            mean, sd = float(targets[0].mean()), float(targets[0].std())
            np.testing.assert_allclose(saved["training_target_sd"], sd, rtol=1e-14, atol=1e-14)
            qrc, _, obs, coupling = architecture(seed, record["gamma"], cfg)
            np.testing.assert_array_equal(saved["coupling"], coupling)
            picks = np.unique(np.array([0, len(times[2]) // 2, len(times[2]) - 1]))
            fit_reports = []
            for row in [record["full_qrc"], *record["rows"]]:
                key = "qrc" if row["model"] == "qrc" else f"replay_d{row['window']}"
                np.testing.assert_allclose(saved[key + "_target_mean"], mean, rtol=1e-14, atol=1e-14)
                np.testing.assert_allclose(saved[key + "_target_sd"], sd, rtol=1e-14, atol=1e-14)
                if float(saved[key + "_alpha"]) != row["alpha"]:
                    raise ValueError("Saved alpha and result row disagree")
                maxima["score_recomputation"] = max(maxima["score_recomputation"], check_score(row, saved[key], targets[2], sd))
                counts["scores"] += 1
                if key == "qrc" or row["window"] in ANCHORS:
                    old = next(r for r in cell["rows"] if r["model"] == row["model"] and r["window"] == row["window"])
                    for metric, delta in check_archive(row, old).items():
                        maxima["archive_" + metric] = max(maxima["archive_" + metric], delta)
                    counts["archived_continuous_qrc" if key == "qrc" else "archived_replay_anchors"] += 1
                    full = ([qrc.run(u, obs, n_virtual=cfg.snapshots)[t] for u, t in zip(blocks, times, strict=True)]
                            if key == "qrc" else
                            [replay_features(qrc, u, t, row["window"], obs, cfg.snapshots)
                             for u, t in zip(blocks, times, strict=True)])
                    audit = check_fit(saved, key, row, full, targets)
                    fit_reports.append({"key": key, **audit})
                    maxima["validation_score_recomputation"] = max(maxima["validation_score_recomputation"], audit["validation_score_difference"])
                    for metric in ("ridge_backward_error", "ridge_condition_number"):
                        maxima[metric] = max(maxima[metric], audit[metric])
                    maxima["prediction_reconstruction"] = max(maxima["prediction_reconstruction"], check_prediction(saved, key, full[2]))
                    counts["full_readout_audits"] += 1
                    counts["full_anchor_predictions"] += len(times[2])
                    del full
                if key != "qrc":
                    direct = np.asarray([qrc.run(blocks[2][t-row["window"]+1:t+1], obs,
                                                  n_virtual=cfg.snapshots)[-1] for t in times[2][picks]])
                    original = replay_features(qrc, blocks[2], times[2][picks], row["window"], obs, cfg.snapshots)
                    np.testing.assert_allclose(direct, original, rtol=0, atol=FEATURE_ATOL)
                    maxima["direct_vs_original_features"] = max(maxima["direct_vs_original_features"], float(np.max(np.abs(direct-original))))
                    maxima["prediction_reconstruction"] = max(maxima["prediction_reconstruction"], check_prediction(saved, key, direct, picks))
                    counts["direct_test_predictions"] += len(picks)
            if seed == cfg.seeds[0]:
                trace = parent["traces"]["narma"]
                np.testing.assert_array_equal(times[2][:96], trace["test_indices"])
                np.testing.assert_allclose(targets[2][:96], trace["target"], rtol=1e-14, atol=1e-14)
                np.testing.assert_allclose(saved["qrc"][:96], trace["qrc"], rtol=0, atol=ARCHIVE_ATOL)
                maxima["archive_trace"] = float(np.max(np.abs(saved["qrc"][:96] - trace["qrc"])))
        counts["roots"] += 1
        root_reports.append({"seed": seed, "fit_audits": fit_reports})
        print(f"Independently verified dense NARMA root {seed}", flush=True)
    report = {
        "schema": "narma_dense_replay_independent_verification_v1", "passed": True,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "counts": counts, "maximum_differences": maxima,
        "tolerances": {"archive_absolute": ARCHIVE_ATOL, "feature_absolute": FEATURE_ATOL,
                       "prediction_absolute": PREDICTION_ATOL, "prediction_relative": PREDICTION_RTOL,
                       "validation_score_absolute": FIT_SCORE_ATOL, "score_absolute": 2e-13, "score_relative": 1e-12},
        "checkpoint_sha256": identities, "parent_sha256": digest(PARENT),
        "verifier_sha256": digest(__file__), "sources": spec["sources"], "root_reports": root_reports,
        "scope": {"all_raw_inputs_targets_and_scores": True,
                  "full_train_validation_test_legacy_replay_windows": list(ANCHORS),
                  "full_continuous_qrc_readouts": True,
                  "direct_evolution_windows": list(WINDOWS),
                  "direct_test_row_positions": "first, middle, last scored row of every root",
                  "training_only_standardization_and_validation_only_alpha_audited": True,
                  "new_loss_selection_performed": False,
                  "same_engine_tolerances_separate_from_archived_runtime_tolerance": True},
    }
    output = results / "independent_verification.json" if output is None else Path(output)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.results, args.output)
    print(json.dumps({"passed": result["passed"], "counts": result["counts"],
                      "maximum_differences": result["maximum_differences"]}, indent=2))
