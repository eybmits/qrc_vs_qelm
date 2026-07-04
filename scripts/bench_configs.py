#!/usr/bin/env python
"""WP6 timing pilot for sizing later frozen row caps."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from qrc_vs_qelm import pipeline, task_families
from qrc_vs_qelm.core import operators, readout, reservoirs, sparse_evolve
from qrc_vs_qelm.matching import recommend_T_evolve
from qrc_vs_qelm.models.api import (
    QELMConfig,
    QRCConfig,
    build_features,
    quantize_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results_freeze"
OUT_PATH = RESULTS_DIR / "timing_pilot.json"
QUANTIZE_LEVELS = (33, 65, 129, 201)
ALPHAS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _machine_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def _inputs(seed: int, T: int, *, binary: bool = False) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if binary:
        return rng.integers(0, 2, size=int(T)).astype(float)
    return rng.uniform(0.02, 0.98, size=int(T))


def _sparse_cd_features(
    inputs: np.ndarray,
    *,
    n_qubits: int,
    coupling_seed: int,
    h: float = 0.5,
    dt: float = 0.5,
    gamma: float = 0.3,
    j_scale: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(int(coupling_seed))
    J = reservoirs.random_couplings(n_qubits, j_scale, rng)
    h0 = reservoirs.ising_xx_hamiltonian(J, h, n_qubits)
    hx = reservoirs.transverse_drive(n_qubits)
    jumps = [(operators.sminus(i, n_qubits), float(gamma)) for i in range(n_qubits)]
    observables = readout.pauli_observables(n_qubits, axes=("x", "y", "z"), max_weight=2)
    return sparse_evolve.run_sparse_lindblad_trajectory(
        n_qubits,
        h0 + h * hx,
        h * hx,
        jumps,
        dt,
        inputs,
        observables,
        washout=0,
        n_virtual=1,
    )


def _time_call(fn: Callable[[], Any]) -> float:
    start = time.perf_counter()
    fn()
    return float(time.perf_counter() - start)


def _class_builders() -> dict[str, dict[str, Any]]:
    seed = 314159
    qelm_base = QELMConfig(
        d=5,
        T_evolve=0.1,
        n_snapshots=1,
        n_qubits=5,
        J_s=1.0,
        h=0.5,
        gamma=0.3,
        dissipator="local_loss",
        quantize_levels=None,
    )
    qelm_rec = recommend_T_evolve(qelm_base, seed)
    qelm_cfg = replace(qelm_base, T_evolve=float(qelm_rec["t_recommended"]))

    return {
        "qrc_n4": {
            "backend": "dense_api",
            "T": 16,
            "builder": lambda: build_features(
                QRCConfig(
                    dt=0.5,
                    h_in=0.5,
                    n_virtual=1,
                    n_qubits=4,
                    J_s=1.0,
                    h=0.5,
                    gamma=0.3,
                    dissipator="local_loss",
                    quantize_levels=33,
                ),
                _inputs(101, 16),
                seed,
            ),
        },
        "qrc_n5": {
            "backend": "dense_api_no_cache",
            "T": 8,
            "builder": lambda: build_features(
                QRCConfig(
                    dt=0.5,
                    h_in=0.5,
                    n_virtual=1,
                    n_qubits=5,
                    J_s=1.0,
                    h=0.5,
                    gamma=0.3,
                    dissipator="local_loss",
                    quantize_levels=None,
                ),
                _inputs(102, 8),
                seed,
            ),
        },
        "qelm_n5": {
            "backend": "dense_qelm_channel",
            "T": 24,
            "builder": lambda: build_features(qelm_cfg, _inputs(103, 24), seed),
            "qelm_recommendation": qelm_rec,
        },
        "qrc_n6_sparse": {
            "backend": "sparse_cd",
            "T": 8,
            "builder": lambda: _sparse_cd_features(_inputs(104, 8), n_qubits=6, coupling_seed=seed),
        },
        "parity_n5": {
            "backend": "sparse_cd_binary_input",
            "T": 32,
            "builder": lambda: _sparse_cd_features(
                _inputs(105, 32, binary=True),
                n_qubits=5,
                coupling_seed=seed,
            ),
        },
    }


def measure_config_classes(repeats: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, spec in _class_builders().items():
        durations = [_time_call(spec["builder"]) for _ in range(int(repeats))]
        record = {
            "median_s": float(statistics.median(durations)),
            "p90_s": _percentile(durations, 90),
            "durations_s": durations,
            "backend": spec["backend"],
            "T": spec["T"],
        }
        if "qelm_recommendation" in spec:
            record["qelm_recommendation"] = spec["qelm_recommendation"]
        out[name] = record
    return out


def _dense_qrc_one_level_seconds(n_qubits: int, levels: int) -> float:
    cfg = QRCConfig(
        dt=0.5,
        h_in=0.5,
        n_virtual=1,
        n_qubits=n_qubits,
        J_s=1.0,
        h=0.5,
        gamma=0.3,
        dissipator="local_loss",
        quantize_levels=levels,
    )
    inputs = np.array([0.5], dtype=float)
    return _time_call(lambda: build_features(cfg, inputs, coupling_seed=314159))


def quantize_level_sweep() -> dict[str, Any]:
    """Report dense cache cost scaling without running infeasible N=6 dense grids."""

    sweep: dict[str, Any] = {"levels": list(QUANTIZE_LEVELS), "n5": {}, "n6": {}}
    n5_per_level: list[float] = []
    for levels in QUANTIZE_LEVELS:
        seconds = _dense_qrc_one_level_seconds(5, levels)
        n5_per_level.append(seconds)
        sweep["n5"][str(levels)] = {
            "status": "measured_one_unique_level",
            "n_qubits": 5,
            "quantize_levels": levels,
            "unique_levels_measured": 1,
            "seconds": seconds,
            "projected_full_grid_seconds": seconds * levels,
        }

    median_n5_per_level = float(statistics.median(n5_per_level))
    for levels in QUANTIZE_LEVELS:
        sparse_seconds = _time_call(
            lambda levels=levels: _sparse_cd_features(
                quantize_inputs(np.array([0.5], dtype=float), levels),
                n_qubits=6,
                coupling_seed=314159,
            )
        )
        projected_per_level = median_n5_per_level * (4 ** (6 - 5))
        sweep["n6"][str(levels)] = {
            "status": "projected_dense_from_n5_per_level_plus_sparse_probe",
            "n_qubits": 6,
            "quantize_levels": levels,
            "sparse_one_step_seconds": sparse_seconds,
            "dense_per_level_projected_seconds": projected_per_level,
            "projected_full_grid_seconds": projected_per_level * levels,
            "note": (
                "Default pilot does not execute dense N=6 quantized expm grids; "
                "the projection follows the WP3 cost finding that cache cost "
                "scales with quantize_levels times 4^N."
            ),
        }
    return sweep


def fidelity_check_stm_n5() -> dict[str, Any]:
    input_seed = 20240618
    coupling_seed = 314159
    T = 240
    washout = 60
    delays = list(range(1, 16))
    base_inputs = _inputs(input_seed, T)
    targets = task_families.stm_targets(base_inputs, delays)
    out: dict[str, Any] = {
        "n_qubits": 5,
        "T": T,
        "washout": washout,
        "coupling_seed": coupling_seed,
        "input_seed": input_seed,
        "levels": {},
    }
    for levels in (33, 201):
        driven = quantize_inputs(base_inputs, levels)
        features = _sparse_cd_features(driven, n_qubits=5, coupling_seed=coupling_seed)
        idx = pipeline.valid_indices(T, washout=washout, max_delay=max(delays), targets=targets)
        splits = pipeline.make_splits(idx)
        scores = pipeline.evaluate_feature_matrix(
            features,
            targets,
            splits,
            alphas=ALPHAS,
        )
        out["levels"][str(levels)] = {
            "stm_MC": float(scores["capacity_test"].sum()),
            "mean_nrmse": float(scores["nrmse_test"].mean()),
            "capacity_by_target": {
                str(row["target"]): float(row["capacity_test"])
                for _, row in scores.iterrows()
            },
        }

    coarse = out["levels"]["33"]
    fine = out["levels"]["201"]
    out["delta_33_minus_201"] = {
        "stm_MC": float(coarse["stm_MC"] - fine["stm_MC"]),
        "mean_nrmse": float(coarse["mean_nrmse"] - fine["mean_nrmse"]),
    }
    return out


def _median_from_timing(timing: dict[str, Any], name: str) -> float:
    record = timing.get(name)
    if record is None:
        record = timing.get("config_classes", {}).get(name)
    if record is None:
        raise KeyError(f"missing timing for config class {name!r}")
    if isinstance(record, (int, float)):
        return float(record)
    return float(record["median_s"])


def project_runtime(timing: dict[str, Any], cells: dict[str, int]) -> dict[str, Any]:
    """Project total seconds from per-class median timings and cell counts."""

    by_class: dict[str, Any] = {}
    total = 0.0
    for name, count in cells.items():
        median_s = _median_from_timing(timing, name)
        projected_s = median_s * int(count)
        by_class[name] = {
            "median_s": median_s,
            "cells": int(count),
            "projected_s": float(projected_s),
        }
        total += projected_s
    return {"by_class": by_class, "total_s": float(total), "total_hours": float(total / 3600.0)}


def run_benchmark(repeats: int) -> dict[str, Any]:
    config_classes = measure_config_classes(repeats)
    planned_cells_example = {
        "qrc_n4": 1,
        "qrc_n5": 1,
        "qelm_n5": 1,
        "qrc_n6_sparse": 1,
        "parity_n5": 1,
    }
    timing = {
        "machine": _machine_metadata(),
        "repeats": int(repeats),
        "config_classes": config_classes,
        "quantize_level_sweep": quantize_level_sweep(),
        "fidelity_check_stm_n5": fidelity_check_stm_n5(),
    }
    timing["runtime_projection_example"] = project_runtime(timing, planned_cells_example)
    return timing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    result = run_benchmark(args.repeats)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
