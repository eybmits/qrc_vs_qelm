"""Deterministic every-window extension of the archived 24-root NARMA replay.

Frozen simulation/selection functions are imported without modification. Shared
prefixes reduce computation but every point is separately evaluated and fitted.
The portable source loader does not require local experiment NPZ checkpoints.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import narma_memory
import task_memory_replication as replication
from laser_demonstration import ALPHAS, ROOT, architecture, digest, dump
from laser_forecast import parameter_arrays, score_prediction
from verify_task_memory_replication import check_coverage

PARENT = ROOT / "paper/figures/task_memory/fig04_task_memory_source.json"
DEFAULT_RESULTS = ROOT / "results_narma_dense_replay_v1"
DEFAULT_SOURCE = ROOT / "paper/figures/narma_dense_replay/source.json"
PROTOCOL = ROOT / "pra_extension/NARMA_DENSE_REPLAY_EXTENSION_PROTOCOL.md"
WINDOWS = tuple(range(1, 49))
ANCHORS = (1, 4, 8, 12, 24, 48)
METRIC_RTOL = 1e-8
METRIC_ATOL = 1e-10


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def _portable(value):
    return json.loads(json.dumps(value, allow_nan=False))


def array_digest(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True)
    return hashlib.sha256(header.encode() + b"\n" + array.tobytes()).hexdigest()


def load_parent():
    parent = json.loads(PARENT.read_text())
    for relative, expected in parent["sources"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Changed archived dependency: {relative}")
    cfg = replication.configs()["narma"]
    summary = parent["study_summaries"]["narma"]
    if summary["config"] != _portable(asdict(cfg)):
        raise ValueError("Archived NARMA configuration changed")
    if summary["config"] != parent["replication_specification"]["configurations"]["narma"]:
        raise ValueError("Archived NARMA specification mismatch")
    check_coverage(summary, "narma", cfg)
    if not parent["independent_verification"]["verification_passed"]:
        raise ValueError("Parent cohort was not independently verified")
    return parent, cfg


def replay_feature_grid(qrc, block, indices, max_window, observables, snapshots):
    """Yield exact replay features for each d, reusing trajectories by start.

    Before length d, each state has seen d-1 full steps from its own fixed
    initial state. Readout maps cover the final step's original checkpoints;
    a full-step propagator then advances that same start for length d+1.
    """
    block = np.asarray(block, float)
    indices = np.asarray(indices, int)
    if (block.ndim != 1 or indices.ndim != 1 or len(indices) == 0
            or max_window < 1 or snapshots < 1 or np.any(np.diff(indices) <= 0)
            or indices[0] < max_window - 1 or indices[-1] >= len(block)
            or not np.isfinite(block).all()):
        raise ValueError("Invalid replay window, scored indices, or input block")
    first_start = int(indices[0]) - max_window + 1
    starts = np.arange(first_start, int(indices[-1]) + 1)
    initial = qrc.initial_state().reshape(-1, order="F")
    states = np.broadcast_to(initial, (len(starts), len(initial))).copy()
    measure = np.stack([o.matrix.T.reshape(-1, order="F") for o in observables])
    width = snapshots * len(observables)
    maps = {}
    for value in np.unique(block[first_start:indices[-1] + 1]):
        substep = qrc._sub_propagator(float(value), qrc.dt / snapshots)
        power, checkpoint_maps = np.eye(len(initial), dtype=complex), []
        for _ in range(snapshots):
            power = substep @ power
            checkpoint_maps.append(measure @ power)
        maps[float(value)] = (np.vstack(checkpoint_maps),
                             qrc._sub_propagator(float(value), qrc.dt))
    for window in range(1, max_window + 1):
        active = len(starts) - window + 1
        states = states[:active]
        endpoints = starts[:active] + window - 1
        values = block[endpoints]
        outputs = np.empty((active, width), float)
        for value in np.unique(values):
            picked = values == value
            current = states[picked]
            readout_map, full_step = maps[float(value)]
            outputs[picked] = np.real(current @ readout_map.T)
            if window < max_window:
                states[picked] = current @ full_step.T
        positions = indices - (first_start + window - 1)
        yield window, outputs[positions]


def compare_anchor(row, archived):
    for key in ("model", "gamma", "window", "features", "alpha"):
        if row[key] != archived[key]:
            raise ValueError(f"Archived {key} differs at d={row['window']}")
    differences = {}
    for key in ("validation_mse", "rmse", "nrmse"):
        np.testing.assert_allclose(row[key], archived[key], rtol=METRIC_RTOL, atol=METRIC_ATOL,
                                   err_msg=f"Archived {key} at d={row['window']}")
        differences[key] = abs(float(row[key]) - float(archived[key]))
    return differences


def _sources(parent):
    return dict(parent["sources"]) | {
        _relative(Path(__file__)): digest(Path(__file__)),
        _relative(PROTOCOL): digest(PROTOCOL),
    }


def initialize(results, parent, cfg):
    results.mkdir(parents=True, exist_ok=True)
    path = results / "extension_specification.json"
    stable = {
        "schema": "narma_dense_replay_extension_v1", "parent_ledger": _relative(PARENT),
        "parent_sha256": digest(PARENT), "config": _portable(asdict(cfg)),
        "windows": list(WINDOWS), "anchor_windows": list(ANCHORS),
        "roots": list(cfg.seeds), "alphas": list(ALPHAS), "sources": _sources(parent),
        "selected_qrc_gamma": {str(c["seed"]): c["selection"]["qrc"]
                               for c in parent["study_summaries"]["narma"]["results"]},
        "metric_tolerance": {"rtol": METRIC_RTOL, "atol": METRIC_ATOL},
        "scope": "Post hoc every-window extension on the unchanged archived 24-root cohort; not a fresh replication or new model/loss selection.",
    }
    if path.exists():
        old = json.loads(path.read_text())
        if {k: v for k, v in old.items() if k != "specified_at_utc"} != stable:
            raise ValueError("Extension specification drift; do not overwrite existing results")
        return old
    spec = stable | {"specified_at_utc": datetime.now(timezone.utc).isoformat()}
    dump(path, spec)
    return spec


def evaluate_root(cell, cfg, results, specification_sha256, parent):
    started = time.perf_counter()
    seed = cell["seed"]
    raw, blocks, indices, targets = narma_memory.make_data(seed, cfg)
    if [len(x) for x in indices] != cell["scored_rows"]:
        raise ValueError("Scored row alignment changed")
    qrc, _, observables, coupling = architecture(seed, cell["selection"]["qrc"], cfg)
    if cfg.snapshots * len(observables) != 48:
        raise ValueError("Replay feature budget changed")
    arrays = {"test_indices": indices[2], "test_target": targets[2], "coupling": coupling,
              "training_target_sd": np.std(targets[0])}
    for label, u, target in zip(("train", "validation", "test"), raw, targets, strict=True):
        arrays[f"{label}_raw_input"] = u
        arrays[f"{label}_target"] = target

    # Same selected continuous QRC, reconstructed solely as an identity check.
    full = [qrc.run(u, observables, n_virtual=cfg.snapshots)[t]
            for u, t in zip(blocks, indices, strict=True)]
    full_row, fitted = narma_memory.fitted_row("qrc", full, targets,
                                             rate=cell["selection"]["qrc"], window=None)
    full_delta = compare_anchor(full_row, next(r for r in cell["rows"] if r["model"] == "qrc"))
    arrays["qrc"] = fitted[3]
    arrays.update(parameter_arrays("qrc", *fitted[:3]))
    if seed == cfg.seeds[0]:
        trace = parent["traces"]["narma"]
        np.testing.assert_array_equal(indices[2][:96], trace["test_indices"])
        np.testing.assert_allclose(targets[2][:96], trace["target"], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(fitted[3][:96], trace["qrc"], rtol=METRIC_RTOL, atol=METRIC_ATOL)
    del full
    generators = [replay_feature_grid(qrc, u, t, max(WINDOWS), observables, cfg.snapshots)
                  for u, t in zip(blocks, indices, strict=True)]
    rows, anchors = [], []
    for window in WINDOWS:
        parts = [next(generator) for generator in generators]
        if any(d != window for d, _ in parts):
            raise ValueError("Window generation out of order")
        row, fit = narma_memory.fitted_row("replay", [x for _, x in parts], targets,
                                          rate=cell["selection"]["qrc"], window=window)
        row["seed"] = seed
        rows.append(row)
        key = f"replay_d{window}"
        arrays[key] = fit[3]
        arrays.update(parameter_arrays(key, *fit[:3]))
        if window in ANCHORS:
            archived = next(r for r in cell["rows"] if r["model"] == "replay" and r["window"] == window)
            anchors.append({"window": window, "differences": compare_anchor(row, archived)})
    filename = f"seed{seed}.npz"
    np.savez_compressed(results / filename, **arrays)
    record = {"seed": seed, "gamma": cell["selection"]["qrc"], "rows": rows,
              "scored_rows": [len(t) for t in indices], "array_file": filename,
              "array_sha256": digest(results / filename), "full_qrc": full_row,
              "full_qrc_differences": full_delta, "anchor_checks": anchors,
              "specification_sha256": specification_sha256,
              "input_sha256": {label: array_digest(u) for label, u in
                               zip(("train", "validation", "test"), raw, strict=True)},
              "seconds": time.perf_counter() - started}
    dump(results / f"seed{seed}.json", record)
    print(json.dumps({"completed_seed": seed, "seconds": record["seconds"],
                      "windows": len(rows), "archived_anchors_verified": len(anchors)}), flush=True)
    return record


def aggregate_rows(rows):
    expected = {(seed, d) for seed in replication.NARMA_ROOTS for d in WINDOWS}
    keys = [(r["seed"], r["window"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected complete 24-root by 48-window grid")
    output = []
    for window in WINDOWS:
        scores = np.asarray([r["nrmse"] for r in rows if r["window"] == window])
        if not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("Invalid NRMSE scores")
        low, med, high = np.quantile(scores, [.1, .5, .9])
        output.append({"window": window, "n": len(scores), "median": float(med),
                       "p10": float(low), "p90": float(high)})
    return output


def _csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish(results, source=DEFAULT_SOURCE):
    parent, cfg = load_parent()
    results, source = Path(results), Path(source)
    spec = initialize(results, parent, cfg)
    records = []
    for seed in cfg.seeds:
        record = json.loads((results / f"seed{seed}.json").read_text())
        if (record["specification_sha256"] != digest(results / "extension_specification.json")
                or digest(results / record["array_file"]) != record["array_sha256"]):
            raise ValueError("Root checkpoint identity mismatch")
        records.append(record)
    rows = [row for record in records for row in record["rows"]]
    aggregates = aggregate_rows(rows)
    source.parent.mkdir(parents=True, exist_ok=True)
    _csv(source.parent / "rows.csv", rows)
    _csv(source.parent / "aggregates.csv", aggregates)
    verification = {
        "passed": True, "completed_roots": len(records), "completed_windows": len(WINDOWS),
        "archived_window_comparisons": sum(len(c["anchor_checks"]) for c in records),
        "continuous_qrc_comparisons": len(records), "first_display_trace_checked": True,
        "anchor_checks": [{"seed": c["seed"], "checks": c["anchor_checks"]} for c in records],
        "continuous_qrc_checks": [{"seed": c["seed"], "differences": c["full_qrc_differences"]}
                                  for c in records],
        "metric_tolerance": spec["metric_tolerance"],
    }
    independent = results / "independent_verification.json"
    if independent.exists():
        verification["independent"] = json.loads(independent.read_text())
    payload = {
        "schema": "narma_dense_replay_v1", "parent_ledger": _relative(PARENT),
        "parent_sha256": digest(PARENT), "specification": spec,
        "specification_sha256": digest(results / "extension_specification.json"),
        "sources": spec["sources"], "rows": rows, "aggregates": aggregates,
        "verification": verification, "scope": spec["scope"],
        "metric": "Test RMSE divided by training-target standard deviation",
        "interval": "10th-90th percentiles of 24 root-level scores; not confidence intervals",
        "algorithm": "Shared chronological prefixes of independent fixed-state replays; one fitted readout per root and window; no interpolation, smoothing, or new loss-rate selection.",
        "source_artifacts": {name: digest(source.parent / name) for name in ("rows.csv", "aggregates.csv")},
        "result_artifacts": {_relative(path): digest(path) for path in sorted(results.iterdir())
                             if path.suffix in (".json", ".npz")},
    }
    validate_payload(payload, parent)
    dump(source, payload)
    return payload


def validate_payload(payload, parent=None):
    if payload.get("schema") != "narma_dense_replay_v1":
        raise ValueError("Unknown dense-replay source schema")
    parent = parent if parent is not None else load_parent()[0]
    if payload["parent_sha256"] != digest(PARENT) or payload["parent_ledger"] != _relative(PARENT):
        raise ValueError("Dense replay identifies a different parent")
    if payload["aggregates"] != aggregate_rows(payload["rows"]):
        raise ValueError("Dense-replay aggregates differ from complete root scores")
    if not payload["verification"]["passed"]:
        raise ValueError("Dense replay verification failed")
    cells = {c["seed"]: c for c in parent["study_summaries"]["narma"]["results"]}
    for row in payload["rows"]:
        if (row["gamma"] != cells[row["seed"]]["selection"]["qrc"]
                or row["alpha"] not in ALPHAS or row["features"] != 48 or row["model"] != "replay"):
            raise ValueError("Dense replay changed model, selection, or feature budget")
        if not np.isfinite([row[k] for k in ("validation_mse", "rmse", "nrmse")]).all():
            raise ValueError("Dense replay contains nonfinite metrics")
        if row["window"] in ANCHORS:
            old = next(r for r in cells[row["seed"]]["rows"]
                       if r["model"] == "replay" and r["window"] == row["window"])
            compare_anchor(row, old)
    if (payload["verification"]["archived_window_comparisons"] != 144
            or payload["verification"]["continuous_qrc_comparisons"] != 24):
        raise ValueError("Incomplete archived-cohort verification")


def load_source(source=DEFAULT_SOURCE):
    """Validate the portable export; local raw checkpoints are not required."""
    source = Path(source)
    payload = json.loads(source.read_text())
    validate_payload(payload)
    for relative, expected in payload["sources"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Changed dense replay dependency: {relative}")
    for name, expected in payload["source_artifacts"].items():
        if digest(source.parent / name) != expected:
            raise ValueError(f"Changed dense replay export: {name}")
    return payload


def run(results=DEFAULT_RESULTS, source=DEFAULT_SOURCE, seeds=None):
    parent, cfg = load_parent()
    results = Path(results)
    spec = initialize(results, parent, cfg)
    specification_sha256 = digest(results / "extension_specification.json")
    cells = {c["seed"]: c for c in parent["study_summaries"]["narma"]["results"]}
    requested = tuple(cfg.seeds if seeds is None else seeds)
    if len(set(requested)) != len(requested) or not set(requested) <= set(cfg.seeds):
        raise ValueError("Requested roots must belong to the unchanged archived cohort")
    for seed in requested:
        checkpoint = results / f"seed{seed}.json"
        if checkpoint.exists():
            recorded = json.loads(checkpoint.read_text())
            if (recorded["specification_sha256"] != specification_sha256
                    or digest(results / recorded["array_file"]) != recorded["array_sha256"]):
                raise ValueError("Cannot resume changed root checkpoint")
            print(json.dumps({"reused_seed": seed}), flush=True)
            continue
        evaluate_root(cells[seed], cfg, results, specification_sha256, parent)
    if all((results / f"seed{s}.json").exists() for s in cfg.seeds):
        return publish(results, source)
    return {"status": "incomplete", "specified_windows": spec["windows"],
            "completed_roots": sum((results / f"seed{s}.json").exists() for s in cfg.seeds)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    payload = publish(args.results, args.source) if args.publish_only else run(args.results, args.source, args.seeds)
    print(json.dumps({"status": payload.get("status", "complete"),
                      "rows": len(payload.get("rows", [])), "source": str(args.source)}), flush=True)
