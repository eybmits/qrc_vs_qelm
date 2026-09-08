"""Frozen, separately seeded feature-density control; no historical refits."""

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

import run_training_data_control as base
from analyze_training_data_control import paired_summary, holm_adjust
from dense_phase_n4_common import prior_seed_registry
from freeze_training_data_control import validate_junit
from long_context_qelm_backend import efficient_student_channel
from qrc_vs_qelm.canonical_emulation import (
    suffix_risk_decomposition,
    degree_two_delay_features,
)

ROOT = base.ROOT
CONFIG = ROOT / "configs/feature_density.json"
FREEZE = ROOT / "configs/feature_density_freeze.json"
ROLES = ("teacher", "student", "train", "val", "test", "prefix", "suffix")


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def read_config(path=CONFIG):
    cfg = json.loads(Path(path).read_text(), object_pairs_hook=base._unique_json_pairs)
    if cfg["teacher_sizes"] != [2, 3, 4] or cfg["checkpoint_counts"] != [
        1,
        2,
        3,
        4,
        6,
        8,
    ]:
        raise ValueError("unregistered size or checkpoint grid")
    if cfg["feature_budgets"] != [18 * m for m in cfg["checkpoint_counts"]]:
        raise ValueError("feature counts do not match local XYZ banks")
    if len(cfg["root_seeds"]) != 12 or len(set(cfg["root_seeds"])) != 12:
        raise ValueError("twelve distinct root seeds required")
    if cfg["window"] > cfg["student_qubits"] or cfg["checkpoint_grid"] != 24:
        raise ValueError("window/register or fine time grid mismatch")
    return cfg


def ledger(cfg, smoke=False):
    seeds = [cfg["development_seed"]] if smoke else cfg["root_seeds"]
    sizes = [2] if smoke else cfg["teacher_sizes"]
    return [
        {
            "root_id": "development" if smoke else f"feature{i + 1:02d}",
            "teacher_n_qubits": n,
            "role": role,
            "entropy": [seed, n, j],
            "seed": int(
                np.random.SeedSequence([seed, n, j]).generate_state(1, dtype=np.uint64)[
                    0
                ]
            ),
        }
        for i, seed in enumerate(seeds)
        for n in sizes
        for j, role in enumerate(ROLES)
    ]


def material(config=CONFIG):
    paths = set((ROOT / "src/qrc_vs_qelm").rglob("*.py"))
    paths.update(
        ROOT / p
        for p in (
            "pyproject.toml",
            "pra_extension/FEATURE_DENSITY_PROTOCOL.md",
            "pra_extension/code/run_feature_density.py",
            "pra_extension/tests/test_feature_density.py",
            "pra_extension/code/run_training_data_control.py",
            "pra_extension/code/analyze_training_data_control.py",
            "pra_extension/code/freeze_training_data_control.py",
            "scripts/long_context_qelm_backend.py",
            "scripts/dense_phase_n4_common.py",
        )
    )
    paths.add(Path(config).resolve())
    return {
        str(p.relative_to(ROOT))
        if p.is_relative_to(ROOT)
        else str(p): base.config_file_sha256(p)
        for p in sorted(paths)
    }


def verify(config=CONFIG, freeze=FREEZE):
    record = json.loads(Path(freeze).read_text())
    if record["status"] != "FROZEN" or record["material"] != material(config):
        raise ValueError("frozen numerical material changed")
    return record


def subset(features, checkpoints, n_obs, fine=24):
    if checkpoints not in (1, 2, 3, 4, 6, 8) or fine % checkpoints:
        raise ValueError("unsupported uniform schedule")
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != fine * n_obs:
        raise ValueError("fine feature bank shape mismatch")
    indices = np.arange(fine // checkpoints - 1, fine, fine // checkpoints)
    return x.reshape(len(x), fine, n_obs)[:, indices, :].reshape(
        len(x), checkpoints * n_obs
    )


def run_cell(cfg, root, n, seeds, smoke=False):
    started = perf_counter()
    anchor = {"student_n_qubits": 6, "window": 6}
    profile = base._canonical_profile({"max_snapshots": 24, "input_levels": 33}, anchor)
    counts = (
        {"train": 32, "val": 16, "test": 24}
        if smoke
        else {k: cfg[k + "_rows"] for k in ("train", "val", "test")}
    )
    burnin = 8 if smoke else cfg["prefix_length"]
    groups = 4 if smoke else cfg["panel_groups"]
    teacher = base.canonical._teacher_reservoir(
        profile,
        n_qubits=n,
        dissipator="local_loss",
        gamma=cfg["gamma"],
        coupling_seed=seeds["teacher"],
    )
    obs = base.canonical._teacher_observables(profile, n)
    channel = efficient_student_channel(profile, seeds["student"])
    student_obs = base.canonical._student_observables(profile)
    inputs = {
        k: base.canonical._inputs(seeds[k], burnin + count, cfg["input_levels"])
        for k, count in counts.items()
    }
    delays = base.canonical._delay_blocks(inputs, window=cfg["window"], burnin=burnin)
    targets = base.canonical._teacher_blocks(teacher, inputs, obs, burnin=burnin)
    panel_profile = {
        "suffix_groups": groups,
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
    windows = {**delays, "panel": suffix[:, ::-1]}
    full = {k: channel.features(v, student_obs) for k, v in windows.items()}
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
    models = [
        (
            "ptd_parallel",
            m,
            {k: subset(v, m, len(student_obs)) for k, v in full.items()},
        )
        for m in cfg["checkpoint_counts"]
    ]
    models.append(
        (
            "poly2_delay",
            0,
            {k: degree_two_delay_features(v) for k, v in windows.items()},
        )
    )
    rows = []
    for model, count, features in models:
        fitted = base.fit_curve_row(
            features,
            targets,
            counts["train"],
            base.canonical._float_grid(cfg["alpha_grid"]),
        )
        prediction = fitted.predict(features["panel"])
        dec = suffix_risk_decomposition(panel_targets, prediction)
        if dec.target_variance <= 1e-8 or abs(dec.identity_error) > 1e-10:
            raise ValueError("invalid normalization or decomposition")
        singular, rank = base.canonical._resolved_singular_values(features["train"])
        row = {
            "root_id": root,
            "teacher_n_qubits": n,
            "student_n_qubits": 6,
            "teacher_gamma": cfg["gamma"],
            "window": 6,
            "model": model,
            "snapshot_budget": count,
            "feature_budget": features["train"].shape[1],
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
        rows.append(row)
    return rows, {
        "root_id": root,
        "teacher_n_qubits": n,
        "seeds": seeds,
        "array_sha256": {k: base.array_sha256(v) for k, v in arrays.items()},
        "seconds": perf_counter() - started,
    }


def coverage(frame, smoke=False):
    roots = ["development"] if smoke else [f"feature{i:02d}" for i in range(1, 13)]
    sizes = [2] if smoke else [2, 3, 4]
    wanted = {
        (r, n, m, b)
        for r in roots
        for n in sizes
        for m, b in [("ptd_parallel", b) for b in (18, 36, 54, 72, 108, 144)]
        + [("poly2_delay", 27)]
    }
    cols = ["root_id", "teacher_n_qubits", "model", "feature_budget"]
    if (
        set(frame[cols].itertuples(index=False, name=None)) != wanted
        or frame.duplicated(cols).any()
    ):
        raise ValueError("incomplete or duplicate coverage")
    if (
        not np.isfinite(frame[["total_mse", "target_variance", "total_nrmse"]])
        .all()
        .all()
    ):
        raise ValueError("nonfinite metric")
    for _, g in frame.groupby(["root_id", "teacher_n_qubits"]):
        if g.target_variance.nunique() != 1 or g.history_mse.nunique() != 1:
            raise ValueError("budgets do not share panel normalization")
    return {"rows": len(frame), "passed": True}


def summarize(frame, cfg):
    contrasts = []
    for n in cfg["teacher_sizes"]:
        q = frame[
            (frame.teacher_n_qubits == n) & (frame.model == "ptd_parallel")
        ].copy()
        q["nmse"] = q.total_mse / q.target_variance
        matrix = q.pivot(
            index="root_id", columns="feature_budget", values="nmse"
        ).sort_index()
        for low, high in cfg["primary_contrasts"]:
            out = paired_summary(
                matrix[low].to_numpy(),
                matrix[high].to_numpy(),
                seed=cfg["analysis_seed"],
                replicates=cfg["bootstrap_replicates"],
            )
            contrasts.append(
                {"teacher_n_qubits": n, "low_budget": low, "high_budget": high, **out}
            )
    for row, p in zip(
        contrasts, holm_adjust([r["one_sided_sign_p"] for r in contrasts]), strict=True
    ):
        row["holm_p_six_contrasts"] = p
    return {
        "contrasts": contrasts,
        "scope": "fixed-model follow-up, not pure approximation or advantage",
    }


def run(output, smoke=False, config=CONFIG, freeze=FREEZE):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output must be empty")
    cfg = read_config(config)
    frozen = None if smoke else verify(config, freeze)
    bound = material(config)
    entries = ledger(cfg, smoke)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "started.json", {"smoke": smoke, "material": bound})
    rows, records = [], []
    started = perf_counter()
    try:
        cells = sorted({(e["root_id"], e["teacher_n_qubits"]) for e in entries})
        for root, n in cells:
            print(f"{root}: N={n}", flush=True)
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
        if material(config) != bound:
            raise ValueError("source changed while running")
        frame.to_csv(output / "metrics.csv", index=False, float_format="%.17g")
        scientific = {
            "arrays": [r["array_sha256"] for r in records],
            "metrics": base.config_file_sha256(output / "metrics.csv"),
        }
        manifest = {
            "status": "COMPLETED",
            "smoke": smoke,
            "material": bound,
            "freeze": frozen,
            "coverage": checked,
            "seed_ledger": entries,
            "cells": records,
            "metrics_sha256": scientific["metrics"],
            "scientific_sha256": sha256(base._json_bytes(scientific)).hexdigest(),
            "seconds": perf_counter() - started,
        }
        dump(output / "manifest.json", manifest)
        if not smoke:
            dump(output / "summary.json", summarize(frame, cfg))
        print(json.dumps(checked), flush=True)
    except Exception as exc:
        dump(output / "failure.json", {"error": str(exc), "smoke": smoke})
        raise


def create_freeze(smoke_a, smoke_b, junit, config=CONFIG, output=FREEZE):
    if Path(output).exists():
        raise FileExistsError("freeze already exists")
    cfg = read_config(config)
    bound = material(config)
    tests = validate_junit(Path(junit))
    records = []
    for directory in (Path(smoke_a), Path(smoke_b)):
        m = json.loads((directory / "manifest.json").read_text())
        if not m["smoke"] or m["status"] != "COMPLETED" or m["material"] != bound:
            raise ValueError("smoke provenance mismatch")
        if m["metrics_sha256"] != base.config_file_sha256(directory / "metrics.csv"):
            raise ValueError("smoke CSV changed")
        coverage(pd.read_csv(directory / "metrics.csv"), True)
        records.append(m)
    if (
        Path(smoke_a).resolve() == Path(smoke_b).resolve()
        or records[0]["scientific_sha256"] != records[1]["scientific_sha256"]
    ):
        raise ValueError("independent identical smokes required")
    for name in ("started.json", "failure.json", "manifest.json"):
        for path in ROOT.glob(f"results_feature_density_*/{name}"):
            if not json.loads(path.read_text()).get("smoke", False):
                raise ValueError("production already opened")
    entries = ledger(cfg) + ledger(cfg, True)
    seeds = (
        cfg["root_seeds"]
        + [cfg["development_seed"], cfg["analysis_seed"]]
        + [e["seed"] for e in entries]
    )
    prior = prior_seed_registry(Path(config))
    if len(set(seeds)) != len(seeds) or any(str(s) in prior["entries"] for s in seeds):
        raise ValueError("seed collision")
    record = {
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "local prospective follow-up, not public preregistration",
        "material": bound,
        "seed_ledger": entries,
        "prior_seed_count": prior["seed_count"],
        "prior_seed_sources": prior["source_files"],
        "tests": tests,
        "smoke_scientific_sha256": records[0]["scientific_sha256"],
    }
    with Path(output).open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


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
