"""Paired exploratory intermediate feature banks; preserve the frozen parent."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import run_feature_density as parent

base = parent.base
ROOT = parent.ROOT
CONFIG = ROOT / "configs/feature_bridge.json"
FREEZE = ROOT / "configs/feature_bridge_freeze.json"
PARENT = ROOT / "results_feature_density_v1"
RESULTS = ROOT / "results_feature_bridge_v1"
dump = parent.dump


def read_config():
    cfg = json.loads(CONFIG.read_text(), object_pairs_hook=base._unique_json_pairs)
    expected = {
        "budgets": [18, 24, 30, 36],
        "new_budgets": [24, 30],
        "anchor_budgets": [18, 36],
        "student_qubits": 6,
        "checkpoint_grid": 24,
        "midpoint_index": 11,
        "final_index": 23,
        "subset_rule": "axis=(qubit+root_phase+round)%3; root_phase=(root_number-1)%3",
    }
    if any(cfg[k] != v for k, v in expected.items()):
        raise ValueError("unregistered bridge design")
    return cfg


def phase(root):
    if root == "development":
        return 0
    if root not in {f"feature{i:02d}" for i in range(1, 13)}:
        raise ValueError("unknown root")
    return (int(root[-2:]) - 1) % 3


def columns(budget, root):
    if budget not in (18, 24, 30, 36):
        raise ValueError("unsupported feature budget")
    # The backend orders observables X0..X5, Y0..Y5, Z0..Z5.
    middle = sorted(
        {
            ((i + phase(root) + j) % 3) * 6 + i
            for j in range((budget - 18) // 6)
            for i in range(6)
        }
    )
    return np.array([11 * 18 + i for i in middle] + list(range(23 * 18, 24 * 18)))


def subset(full, budget, root):
    if np.asarray(full).ndim != 2 or full.shape[1] != 432:
        raise ValueError("expected the parent's 24 by 18 feature grid")
    if budget in (18, 36):
        return parent.subset(full, budget // 18, 18)
    return np.ascontiguousarray(full[:, columns(budget, root)])


def parent_artifacts(smoke=False):
    parent.verify()
    directory = ROOT / "results_feature_density_smoke_a" if smoke else PARENT
    m = json.loads((directory / "manifest.json").read_text())
    if m["status"] != "COMPLETED" or m["smoke"] != smoke:
        raise ValueError("parent not completed in expected mode")
    if m["material"] != parent.material():
        raise ValueError("parent numerical provenance changed")
    if m["metrics_sha256"] != base.config_file_sha256(directory / "metrics.csv"):
        raise ValueError("parent metrics changed")
    f = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    parent.coverage(f, smoke)
    return f, m


def material():
    paths = [
        CONFIG,
        ROOT / "pra_extension/FEATURE_BRIDGE_PROTOCOL.md",
        Path(__file__),
        ROOT / "pra_extension/tests/test_feature_bridge.py",
        parent.FREEZE,
    ]
    for directory in (PARENT, ROOT / "results_feature_density_smoke_a"):
        paths.extend(directory / name for name in ("metrics.csv", "manifest.json"))
    return {
        **parent.material(),
        **{str(p.relative_to(ROOT)): base.config_file_sha256(p) for p in paths},
    }


def verify():
    record = json.loads(FREEZE.read_text())
    if record["status"] != "FROZEN" or record["material"] != material():
        raise ValueError("bridge numerical material changed")
    return record


def run_cell(cfg, root, n, seeds, smoke=False):
    started = perf_counter()
    profile = base._canonical_profile(
        {"max_snapshots": 24, "input_levels": 33}, {"student_n_qubits": 6, "window": 6}
    )
    counts = (
        {"train": 32, "val": 16, "test": 24}
        if smoke
        else {k: cfg[k + "_rows"] for k in ("train", "val", "test")}
    )
    burnin = 8 if smoke else cfg["prefix_length"]
    teacher = base.canonical._teacher_reservoir(
        profile,
        n_qubits=n,
        dissipator="local_loss",
        gamma=cfg["gamma"],
        coupling_seed=seeds["teacher"],
    )
    obs = base.canonical._teacher_observables(profile, n)
    channel = parent.efficient_student_channel(profile, seeds["student"])
    student_obs = base.canonical._student_observables(profile)
    inputs = {
        k: base.canonical._inputs(seeds[k], burnin + count, cfg["input_levels"])
        for k, count in counts.items()
    }
    delays = base.canonical._delay_blocks(inputs, window=cfg["window"], burnin=burnin)
    targets = base.canonical._teacher_blocks(teacher, inputs, obs, burnin=burnin)
    panel_profile = {
        "suffix_groups": 4 if smoke else cfg["panel_groups"],
        "prefix_branches": 4,
        "prefix_burnin": burnin,
        "quantize_levels": 33,
        "windows": [6],
    }
    prefixes, suffixes = base.canonical._suffix_inputs(
        panel_profile,
        {"prefix_input_seed": seeds["prefix"], "suffix_input_seed": seeds["suffix"]},
    )
    suffix = suffixes[6]
    panel_targets = base.canonical._suffix_targets(
        teacher, base.canonical._prefix_states(teacher, prefixes), suffix, obs
    )
    full = {
        k: channel.features(v, student_obs)
        for k, v in {**delays, "panel": suffix[:, ::-1]}.items()
    }
    arrays = {
        **inputs,
        **{f"targets_{k}": v for k, v in targets.items()},
        "prefixes": prefixes,
        "suffixes_chronological": suffix,
        "panel_targets": panel_targets,
        **{f"features_{k}": v for k, v in full.items()},
    }
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("nonfinite scientific arrays")
    rows = []
    for budget in read_config()["budgets"]:
        features = {k: subset(v, budget, root) for k, v in full.items()}
        fitted = base.fit_curve_row(
            features,
            targets,
            counts["train"],
            base.canonical._float_grid(cfg["alpha_grid"]),
        )
        dec = parent.suffix_risk_decomposition(
            panel_targets, fitted.predict(features["panel"])
        )
        if dec.target_variance <= 1e-8 or abs(dec.identity_error) > 1e-10:
            raise ValueError("invalid normalization or decomposition")
        singular, rank = base.canonical._resolved_singular_values(features["train"])
        rows.append(
            {
                "root_id": root,
                "teacher_n_qubits": n,
                "student_n_qubits": 6,
                "teacher_gamma": cfg["gamma"],
                "window": 6,
                "model": "ptd_parallel",
                "snapshot_budget": 1 if budget == 18 else 2,
                "feature_budget": budget,
                "root_phase": phase(root),
                "midpoint_features": budget - 18,
                "feature_columns": ";".join(map(str, columns(budget, root))),
                "experiment_stage": "exploratory_partial_bank"
                if budget in (24, 30)
                else "anchor_replication",
                "alpha": "inf" if np.isinf(fitted.alpha) else float(fitted.alpha),
                "feature_rank": rank,
                "resolved_condition_number": float(singular[0] / singular[rank - 1])
                if rank
                else None,
                "train_rows": counts["train"],
                "val_rows": counts["val"],
                "fit_rows": counts["train"] + counts["val"],
                "test_rows": counts["test"],
                "feature_mean_sha256": base.array_sha256(fitted.feature_mean),
                "feature_scale_sha256": base.array_sha256(fitted.feature_scale),
                **asdict(dec),
                "total_nrmse": dec.total_nrmse,
                "history_nrmse": dec.history_nrmse,
                "residual_nrmse": dec.representation_nrmse,
                "stream_test_mse": float(
                    np.mean((fitted.predict(features["test"]) - targets["test"]) ** 2)
                ),
            }
        )
    return rows, {
        "root_id": root,
        "teacher_n_qubits": n,
        "seeds": seeds,
        "array_sha256": {k: base.array_sha256(v) for k, v in arrays.items()},
        "seconds": perf_counter() - started,
    }


def validate_anchors(frame, records, smoke=False):
    original, manifest = parent_artifacts(smoke)
    key = ["root_id", "teacher_n_qubits", "feature_budget"]
    left = frame[frame.feature_budget.isin([18, 36])].set_index(key).sort_index()
    right = (
        original[
            (original.model == "ptd_parallel") & original.feature_budget.isin([18, 36])
        ]
        .set_index(key)
        .sort_index()
    )
    if not left.index.equals(right.index):
        raise ValueError("anchor coverage mismatch")
    for c in right.columns:
        if pd.api.types.is_numeric_dtype(right[c]):
            if not np.allclose(
                left[c].astype(float),
                right[c].astype(float),
                rtol=1e-11,
                atol=1e-14,
                equal_nan=True,
            ):
                raise ValueError(f"anchor metric mismatch: {c}")
        elif not left[c].equals(right[c]):
            raise ValueError(f"anchor metadata mismatch: {c}")
    original_arrays = {
        (r["root_id"], r["teacher_n_qubits"]): r["array_sha256"]
        for r in manifest["cells"]
    }
    arrays = {(r["root_id"], r["teacher_n_qubits"]): r["array_sha256"] for r in records}
    if arrays != original_arrays:
        raise ValueError("parent scientific arrays do not match")
    return {
        "passed": True,
        "anchor_rows": len(left),
        "array_cells": len(arrays),
        "rtol": 1e-11,
        "atol": 1e-14,
    }


def coverage(frame, smoke=False):
    roots = ["development"] if smoke else [f"feature{i:02d}" for i in range(1, 13)]
    sizes = [2] if smoke else [2, 3, 4]
    key = ["root_id", "teacher_n_qubits", "feature_budget"]
    expected = {(r, n, b) for r in roots for n in sizes for b in (18, 24, 30, 36)}
    if (
        set(frame[key].itertuples(index=False, name=None)) != expected
        or frame.duplicated(key).any()
    ):
        raise ValueError("incomplete or duplicate bridge coverage")
    if (
        not np.isfinite(frame[["total_mse", "target_variance", "total_nrmse"]])
        .all()
        .all()
    ):
        raise ValueError("nonfinite metrics")
    if not np.allclose(
        frame.total_mse, frame.history_mse + frame.representation_mse, atol=1e-13
    ):
        raise ValueError("risk decomposition mismatch")
    if not np.allclose(
        frame.total_nrmse, np.sqrt(frame.total_mse / frame.target_variance)
    ):
        raise ValueError("normalization mismatch")
    for _, g in frame.groupby(key[:2]):
        if g.history_mse.nunique() != 1 or g.target_variance.nunique() != 1:
            raise ValueError("unpaired panels")
    return {
        "passed": True,
        "rows": len(frame),
        "new_rows": int(frame.feature_budget.isin([24, 30]).sum()),
    }


def summarize(frame):
    medians, paired = [], []
    for (n, b), group in frame.groupby(["teacher_n_qubits", "feature_budget"]):
        medians.append(
            {
                "teacher_n_qubits": int(n),
                "feature_budget": int(b),
                "median_percent": float(100 * group.total_nrmse.median()),
                "q10_percent": float(100 * group.total_nrmse.quantile(0.1)),
                "q90_percent": float(100 * group.total_nrmse.quantile(0.9)),
            }
        )
    for n, group in frame.groupby("teacher_n_qubits"):
        matrix = group.pivot(
            index="root_id", columns="feature_budget", values="total_mse"
        )
        for low, high in ((18, 24), (24, 30), (30, 36)):
            paired.append(
                {
                    "teacher_n_qubits": int(n),
                    "low": low,
                    "high": high,
                    "roots_with_lower_mse": int((matrix[high] < matrix[low]).sum()),
                    "median_relative_mse_reduction": float(
                        (1 - matrix[high] / matrix[low]).median()
                    ),
                }
            )
    return {
        "scope": "descriptive paired extension on existing roots, no confirmatory inference",
        "medians": medians,
        "paired": paired,
    }


def run(output, smoke=False):
    read_config()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output must be empty")
    parent_artifacts(smoke)
    frozen = None if smoke else verify()
    bound, cfg = material(), parent.read_config()
    entries = parent.ledger(cfg, smoke)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "started.json", {"smoke": smoke, "material": bound})
    rows, records, started = [], [], perf_counter()
    try:
        for root, n in sorted({(e["root_id"], e["teacher_n_qubits"]) for e in entries}):
            print(f"{root}: N={n}; B=18,24,30,36", flush=True)
            seeds = {
                e["role"]: e["seed"]
                for e in entries
                if e["root_id"] == root and e["teacher_n_qubits"] == n
            }
            values, record = run_cell(cfg, root, n, seeds, smoke)
            rows.extend(values)
            records.append(record)
        frame = pd.DataFrame(rows)
        checked = coverage(frame, smoke)
        anchors = validate_anchors(frame, records, smoke)
        if material() != bound:
            raise ValueError("source changed while running")
        frame.to_csv(output / "metrics.csv", index=False, float_format="%.17g")
        digest = base.config_file_sha256(output / "metrics.csv")
        scientific = {"arrays": [r["array_sha256"] for r in records], "metrics": digest}
        dump(
            output / "manifest.json",
            {
                "status": "COMPLETED",
                "smoke": smoke,
                "material": bound,
                "freeze": frozen,
                "coverage": checked,
                "anchors": anchors,
                "cells": records,
                "seed_ledger": entries,
                "metrics_sha256": digest,
                "scientific_sha256": sha256(base._json_bytes(scientific)).hexdigest(),
                "seconds": perf_counter() - started,
            },
        )
        if not smoke:
            dump(output / "summary.json", summarize(frame))
        print(json.dumps({**checked, "anchors": anchors}), flush=True)
    except Exception as exc:
        dump(output / "failure.json", {"error": str(exc), "smoke": smoke})
        raise


def create_freeze(smoke_a, smoke_b, junit):
    if FREEZE.exists():
        raise FileExistsError("freeze already exists")
    read_config()
    bound, records = material(), []
    tests = parent.validate_junit(Path(junit))
    for directory in (Path(smoke_a), Path(smoke_b)):
        m = json.loads((directory / "manifest.json").read_text())
        f = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
        if not m["smoke"] or m["status"] != "COMPLETED" or m["material"] != bound:
            raise ValueError("development provenance mismatch")
        if m["metrics_sha256"] != base.config_file_sha256(directory / "metrics.csv"):
            raise ValueError("development CSV changed")
        coverage(f, True)
        validate_anchors(f, m["cells"], True)
        records.append(m)
    if (
        Path(smoke_a).resolve() == Path(smoke_b).resolve()
        or records[0]["scientific_sha256"] != records[1]["scientific_sha256"]
    ):
        raise ValueError("two distinct identical smokes required")
    for path in ROOT.glob("results_feature_bridge_*/started.json"):
        if not json.loads(path.read_text())["smoke"]:
            raise ValueError("production already started")
    dump(
        FREEZE,
        {
            "status": "FROZEN",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "local lock before exploratory intermediate fits; parent roots reused",
            "material": bound,
            "tests": tests,
            "seed_ledger": parent.ledger(parent.read_config()),
            "smoke_scientific_sha256": records[0]["scientific_sha256"],
        },
    )


def load_completed(directory=RESULTS):
    verify()
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    if m["status"] != "COMPLETED" or m["smoke"] or m["material"] != material():
        raise ValueError("invalid bridge provenance")
    if m["metrics_sha256"] != base.config_file_sha256(directory / "metrics.csv"):
        raise ValueError("bridge CSV changed")
    f = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    coverage(f)
    validate_anchors(f, m["cells"])
    return f, m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["smoke", "freeze", "run"])
    p.add_argument("--output", type=Path)
    p.add_argument("--smoke-a", type=Path)
    p.add_argument("--smoke-b", type=Path)
    p.add_argument("--junit", type=Path)
    args = p.parse_args()
    if args.mode == "freeze":
        create_freeze(args.smoke_a, args.smoke_b, args.junit)
    else:
        run(args.output, smoke=args.mode == "smoke")


if __name__ == "__main__":
    main()
