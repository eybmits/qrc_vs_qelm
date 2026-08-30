"""Exploratory QELM architecture-family robustness experiment.

The canonical student is a six-qubit complete-graph transverse-field Ising
QELM.  This module keeps its encoding, register size, evolution time,
snapshots, measurements, ridge-selection rule, and dense coupling draw fixed,
then replaces only the coupling graph by a chain, ring, or star.  Sparse graphs
are rescaled to the exact edge-coefficient L2 norm of the paired dense draw.
Because distinct Pauli XX strings are Hilbert--Schmidt orthogonal, this also
matches the normalized Hilbert--Schmidt norm of the interaction Hamiltonian.

Every model-selection decision uses training/validation rows.  Test targets are
used only once after the ridge penalty and (in the analysis) architecture have
been selected.  The quadratic delay control is retained explicitly: this is an
architecture-robustness check, not a quantum-advantage experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from finite_shots import AXES, exact_single_site_features, qelm_statevectors
from observable_memory import StressConfig, build_reservoir, default_config
from qrc_vs_qelm import canonical_emulation, controls, emulation
from qrc_vs_qelm.core import readout, reservoirs
from qrc_vs_qelm.models import qelm


ARCHITECTURES = ("dense", "chain", "ring", "star")
ARCHITECTURE_SEED_CODES = {name: index + 1 for index, name in enumerate(ARCHITECTURES)}


@dataclass(frozen=True)
class ArchitectureRoot:
    root_id: str
    teacher_coupling_seeds: Mapping[int, int]
    student_coupling_seeds: Mapping[str, int]
    input_seed: int


@dataclass(frozen=True)
class ArchitectureStudyConfig:
    roots: tuple[ArchitectureRoot, ...]
    teacher_n_qubits: tuple[int, ...]
    architectures: tuple[str, ...]
    teacher_gamma: float
    window: int
    train_rows: int
    validation_rows: int
    test_rows: int
    burnin: int
    quantize_levels: int
    snapshot_budgets: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    student_n_qubits: int = 6
    max_snapshots: int = 4
    dt: float = 0.5
    h: float = 0.5
    h_in: float = 1.0
    coupling_scale: float = 1.0


def default_architecture_config(profile: str = "full") -> ArchitectureStudyConfig:
    alphas = (
        0.0,
        1e-10,
        1e-8,
        1e-6,
        1e-4,
        1e-2,
        1.0,
        1e2,
        1e4,
        1e6,
        float("inf"),
    )
    if profile == "smoke":
        roots = (
            ArchitectureRoot(
                root_id="arch_smoke01",
                teacher_coupling_seeds={2: 9_601_012},
                student_coupling_seeds={
                    name: 9_621_010 + code
                    for name, code in ARCHITECTURE_SEED_CODES.items()
                },
                input_seed=9_611_011,
            ),
        )
        return ArchitectureStudyConfig(
            roots=roots,
            teacher_n_qubits=(2,),
            architectures=ARCHITECTURES,
            teacher_gamma=4.0,
            window=6,
            train_rows=32,
            validation_rows=16,
            test_rows=24,
            burnin=16,
            quantize_levels=9,
            snapshot_budgets=(1, 2, 4),
            alpha_grid=alphas,
        )
    if profile != "full":
        raise ValueError("profile must be 'smoke' or 'full'")
    roots = tuple(
        ArchitectureRoot(
            root_id=f"arch{index:02d}",
            teacher_coupling_seeds={
                n: 9_600_000 + 100 * index + n for n in (2, 3, 4)
            },
            student_coupling_seeds={
                name: 9_700_000 + 100 * index + code
                for name, code in ARCHITECTURE_SEED_CODES.items()
            },
            input_seed=9_800_000 + 100 * index + 1,
        )
        for index in range(1, 13)
    )
    return ArchitectureStudyConfig(
        roots=roots,
        teacher_n_qubits=(2, 3, 4),
        architectures=ARCHITECTURES,
        teacher_gamma=4.0,
        window=6,
        train_rows=128,
        validation_rows=48,
        test_rows=96,
        burnin=48,
        quantize_levels=33,
        snapshot_budgets=(1, 2, 4),
        alpha_grid=alphas,
    )


def validate_architecture_config(config: ArchitectureStudyConfig) -> None:
    if not config.roots or not config.teacher_n_qubits:
        raise ValueError("roots and teacher sizes must be nonempty")
    if config.architectures != ARCHITECTURES:
        raise ValueError(f"architectures must equal {ARCHITECTURES}")
    if config.student_n_qubits < config.window:
        raise ValueError("one-to-one encoding requires student_n_qubits >= window")
    if tuple(sorted(set(config.snapshot_budgets))) != config.snapshot_budgets:
        raise ValueError("snapshot budgets must be strictly increasing")
    if max(config.snapshot_budgets) != config.max_snapshots:
        raise ValueError("largest snapshot budget must equal max_snapshots")
    if any(config.max_snapshots % budget for budget in config.snapshot_budgets):
        raise ValueError("snapshot budgets must divide max_snapshots")
    if min(config.train_rows, config.validation_rows, config.test_rows, config.burnin) < 1:
        raise ValueError("row counts and burnin must be positive")
    if config.quantize_levels < 2:
        raise ValueError("quantize_levels must be >= 2")
    if not config.alpha_grid or any(alpha < 0 or np.isnan(alpha) for alpha in config.alpha_grid):
        raise ValueError("invalid alpha grid")
    root_ids = [root.root_id for root in config.roots]
    if len(set(root_ids)) != len(root_ids) or any(not value for value in root_ids):
        raise ValueError("root ids must be nonempty and unique")
    seen: dict[int, str] = {}
    for root in config.roots:
        if set(root.teacher_coupling_seeds) != set(config.teacher_n_qubits):
            raise ValueError(f"teacher seed coverage mismatch for {root.root_id}")
        if tuple(root.student_coupling_seeds) != config.architectures:
            raise ValueError(f"student seed coverage/order mismatch for {root.root_id}")
        entries = {
            **{f"teacher_N{n}": seed for n, seed in root.teacher_coupling_seeds.items()},
            **{f"student_{name}": seed for name, seed in root.student_coupling_seeds.items()},
            "input": root.input_seed,
        }
        for label, raw_seed in entries.items():
            seed = int(raw_seed)
            if seed < 0:
                raise ValueError("seeds must be nonnegative")
            if seed in seen:
                raise ValueError(f"seed collision: {root.root_id}.{label} and {seen[seed]}")
            seen[seed] = f"{root.root_id}.{label}"


def graph_edges(architecture: str, n_qubits: int) -> tuple[tuple[int, int], ...]:
    n = int(n_qubits)
    if architecture == "dense":
        return tuple(combinations(range(n), 2))
    if architecture == "chain":
        return tuple((i, i + 1) for i in range(n - 1))
    if architecture == "ring":
        return tuple((i, i + 1) for i in range(n - 1)) + ((0, n - 1),)
    if architecture == "star":
        return tuple((0, i) for i in range(1, n))
    raise ValueError(f"unknown architecture: {architecture!r}")


def edge_coefficient_l2(couplings: np.ndarray) -> float:
    matrix = np.asarray(couplings, float)
    return float(np.linalg.norm(matrix[np.triu_indices_from(matrix, k=1)]))


def matched_architecture_couplings(
    *,
    root: ArchitectureRoot,
    architecture: str,
    n_qubits: int,
    coupling_scale: float,
    dense_reference_norm: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Return graph-masked couplings, applied scaling, and edge L2 norm.

    The dense branch is byte-for-byte the existing random-coupling construction
    for its declared seed.  Sparse branches are norm matched to the paired dense
    realization without modifying the dense draw.
    """

    rng = np.random.default_rng(int(root.student_coupling_seeds[architecture]))
    raw = reservoirs.random_couplings(int(n_qubits), float(coupling_scale), rng)
    if architecture == "dense":
        norm = edge_coefficient_l2(raw)
        return raw, 1.0, norm
    if dense_reference_norm is None or float(dense_reference_norm) <= 0.0:
        raise ValueError("sparse architectures require a positive dense reference norm")
    mask = np.zeros_like(raw)
    for first, second in graph_edges(architecture, int(n_qubits)):
        mask[first, second] = mask[second, first] = 1.0
    sparse = raw * mask
    raw_norm = edge_coefficient_l2(sparse)
    if raw_norm <= 0.0:
        raise RuntimeError("sampled sparse coupling norm vanished")
    factor = float(dense_reference_norm) / raw_norm
    matched = sparse * factor
    return matched, factor, edge_coefficient_l2(matched)


def _input_stream(config: ArchitectureStudyConfig, root: ArchitectureRoot) -> np.ndarray:
    total_rows = config.train_rows + config.validation_rows + config.test_rows
    total_inputs = config.burnin + config.window - 1 + total_rows
    rng = np.random.default_rng(int(root.input_seed))
    return rng.integers(0, config.quantize_levels, size=total_inputs).astype(float) / (
        config.quantize_levels - 1
    )


def _teacher_targets(
    config: ArchitectureStudyConfig,
    root: ArchitectureRoot,
    inputs: np.ndarray,
    teacher_n_qubits: int,
) -> np.ndarray:
    template = default_config("smoke")
    teacher_config = StressConfig(
        roots=template.roots,
        n_qubits=(int(teacher_n_qubits),),
        dissipator_families=("local_loss",),
        gammas=(float(config.teacher_gamma),),
        windows=(int(config.window),),
        observable_panels=template.observable_panels,
        quantize_levels=int(config.quantize_levels),
        suffix_groups=4,
        prefix_branches=3,
        prefix_burnin=int(config.burnin),
        dt=float(config.dt),
        h=float(config.h),
        h_in=float(config.h_in),
        coupling_scale=float(config.coupling_scale),
    )
    teacher, _ = build_reservoir(
        n_qubits=int(teacher_n_qubits),
        family="local_loss",
        gamma=float(config.teacher_gamma),
        coupling_seed=int(root.teacher_coupling_seeds[int(teacher_n_qubits)]),
        config=teacher_config,
    )
    observables = readout.pauli_observables(
        int(teacher_n_qubits), axes=AXES, max_weight=2
    )
    first = config.burnin + config.window - 1
    return teacher.run(inputs, observables, washout=first, n_virtual=1)


def _student_max_features(
    config: ArchitectureStudyConfig,
    root: ArchitectureRoot,
    inputs: np.ndarray,
    architecture: str,
    dense_reference_norm: float,
) -> tuple[np.ndarray, float, float, int]:
    couplings, scaling, matched_norm = matched_architecture_couplings(
        root=root,
        architecture=architecture,
        n_qubits=config.student_n_qubits,
        coupling_scale=config.coupling_scale,
        dense_reference_norm=dense_reference_norm,
    )
    hamiltonian = reservoirs.ising_xx_hamiltonian(
        couplings, float(config.h), int(config.student_n_qubits)
    )
    channel = qelm.UnitaryQELMChannel.from_hamiltonian(
        n_qubits=int(config.student_n_qubits),
        H_static=hamiltonian,
        T_evolve=float(config.dt),
        n_snapshots=int(config.max_snapshots),
    )
    first = config.burnin + config.window - 1
    windows = qelm.delay_embed(inputs, int(config.window))[first:]
    states = qelm_statevectors(channel, windows)
    features = exact_single_site_features(states, int(config.student_n_qubits))
    return features, scaling, matched_norm, len(graph_edges(architecture, config.student_n_qubits))


def _fit_row(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    alphas: Sequence[float],
) -> dict[str, Any]:
    model = emulation.fit_multivariate_ridge(
        features,
        targets,
        train,
        validation,
        alphas=alphas,
    )
    prediction = model.predict(features[test])
    metrics = emulation.emulation_metrics(targets[test], prediction)
    centered_targets = targets[test] - np.mean(targets[test], axis=0, keepdims=True)
    train_validation = np.concatenate([train, validation])
    centered = features[train_validation] - np.mean(features[train_validation], axis=0)
    return {
        "selected_alpha": "+inf" if np.isposinf(model.alpha) else format(model.alpha, ".12g"),
        "intercept_only_selected": bool(np.isposinf(model.alpha)),
        "validation_nrmse": float(model.validation_nrmse),
        "test_nrmse": float(metrics.nrmse),
        "test_mse": float(metrics.mse),
        "test_target_variance": float(np.mean(centered_targets**2)),
        "feature_rank_train_validation": int(np.linalg.matrix_rank(centered)),
    }


def run_architecture_study(config: ArchitectureStudyConfig) -> pd.DataFrame:
    """Evaluate graph families and the quadratic control on untouched tests."""

    validate_architecture_config(config)
    rows: list[dict[str, Any]] = []
    total_rows = config.train_rows + config.validation_rows + config.test_rows
    train = np.arange(config.train_rows, dtype=int)
    validation = np.arange(config.train_rows, config.train_rows + config.validation_rows)
    test = np.arange(config.train_rows + config.validation_rows, total_rows)
    n_observables = len(AXES) * config.student_n_qubits

    for root in config.roots:
        inputs = _input_stream(config, root)
        first = config.burnin + config.window - 1
        dense, _, dense_norm = matched_architecture_couplings(
            root=root,
            architecture="dense",
            n_qubits=config.student_n_qubits,
            coupling_scale=config.coupling_scale,
        )
        # Defensive audit of exact reuse of the canonical construction.
        reference_rng = np.random.default_rng(int(root.student_coupling_seeds["dense"]))
        reference_dense = reservoirs.random_couplings(
            config.student_n_qubits, config.coupling_scale, reference_rng
        )
        if not np.array_equal(dense, reference_dense):
            raise RuntimeError("dense architecture no longer reproduces canonical construction")

        max_features: dict[str, tuple[np.ndarray, float, float, int]] = {}
        for architecture in config.architectures:
            max_features[architecture] = _student_max_features(
                config, root, inputs, architecture, dense_norm
            )
        control_max = controls.poly_delay_features(
            inputs, d=config.window, D=config.max_snapshots * n_observables
        )[first:]

        for teacher_n in config.teacher_n_qubits:
            targets = _teacher_targets(config, root, inputs, int(teacher_n))
            if targets.shape[0] != total_rows:
                raise RuntimeError("target split construction produced the wrong row count")
            for budget in config.snapshot_budgets:
                width = budget * n_observables
                for architecture in config.architectures:
                    feature_max, scaling, matched_norm, edge_count = max_features[architecture]
                    features = canonical_emulation.select_nested_snapshots(
                        feature_max,
                        max_snapshots=config.max_snapshots,
                        selected_snapshots=budget,
                        n_observables=n_observables,
                    )
                    fit = _fit_row(
                        features=features,
                        targets=targets,
                        train=train,
                        validation=validation,
                        test=test,
                        alphas=config.alpha_grid,
                    )
                    rows.append(
                        {
                            "root_id": root.root_id,
                            "teacher_n_qubits": int(teacher_n),
                            "teacher_gamma": float(config.teacher_gamma),
                            "student_model": "qelm",
                            "student_architecture": architecture,
                            "student_n_qubits": int(config.student_n_qubits),
                            "window": int(config.window),
                            "snapshot_budget": int(budget),
                            "feature_budget": int(features.shape[1]),
                            "graph_edge_count": int(edge_count),
                            "coupling_rescale_factor": float(scaling),
                            "coupling_edge_l2": float(matched_norm),
                            "dense_reference_edge_l2": float(dense_norm),
                            "student_coupling_seed": int(root.student_coupling_seeds[architecture]),
                            "teacher_coupling_seed": int(root.teacher_coupling_seeds[int(teacher_n)]),
                            "input_seed": int(root.input_seed),
                            **fit,
                        }
                    )

                control_features = control_max[:, :width]
                fit = _fit_row(
                    features=control_features,
                    targets=targets,
                    train=train,
                    validation=validation,
                    test=test,
                    alphas=config.alpha_grid,
                )
                rows.append(
                    {
                        "root_id": root.root_id,
                        "teacher_n_qubits": int(teacher_n),
                        "teacher_gamma": float(config.teacher_gamma),
                        "student_model": "quadratic_delay",
                        "student_architecture": "quadratic_delay",
                        "student_n_qubits": 0,
                        "window": int(config.window),
                        "snapshot_budget": int(budget),
                        "feature_budget": int(control_features.shape[1]),
                        "graph_edge_count": 0,
                        "coupling_rescale_factor": 0.0,
                        "coupling_edge_l2": 0.0,
                        "dense_reference_edge_l2": float(dense_norm),
                        "student_coupling_seed": 0,
                        "teacher_coupling_seed": int(root.teacher_coupling_seeds[int(teacher_n)]),
                        "input_seed": int(root.input_seed),
                        **fit,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["root_id", "teacher_n_qubits", "snapshot_budget", "student_architecture"],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_architecture_results(results: pd.DataFrame) -> dict[str, Any]:
    cells = []
    for key, group in results.groupby(
        ["student_architecture", "teacher_n_qubits", "snapshot_budget"], sort=True
    ):
        cells.append(
            {
                "student_architecture": str(key[0]),
                "teacher_n_qubits": int(key[1]),
                "snapshot_budget": int(key[2]),
                "n_roots": int(group.root_id.nunique()),
                "median_test_nrmse": float(np.median(group.test_nrmse)),
                "median_validation_nrmse": float(np.median(group.validation_nrmse)),
            }
        )
    return {
        "schema_version": 1,
        "study": "pra_qelm_student_architecture_family",
        "status": "EXPLORATORY_STRESS_TEST",
        "claim_boundary": (
            "Norm-matched graph-topology robustness for one six-qubit unitary "
            "QELM family; not architecture universality or quantum advantage."
        ),
        "counts": {
            "rows": int(len(results)),
            "roots": int(results.root_id.nunique()),
            "teacher_sizes": sorted(int(value) for value in results.teacher_n_qubits.unique()),
            "architectures": sorted(str(value) for value in results.student_architecture.unique()),
        },
        "cells": cells,
    }
