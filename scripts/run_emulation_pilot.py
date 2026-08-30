#!/usr/bin/env python
"""Exploratory QRC-teacher / reset-QELM-student emulation pilot.

The target is the complete accessible QRC Pauli-output vector.  The study
separates memory truncation (same-coupling reset replay), static feature access
(nested banks of independent QELMs), and a classical linear-delay control.  A
fresh common-suffix panel supplies exact sampled lower-bound certificates for
every deterministic window-only predictor.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrc_vs_qelm import manifests
from qrc_vs_qelm.core import diagnostics, dissipators, readout, reservoirs
from qrc_vs_qelm.emulation import (
    MultiOutputRidge,
    emulation_metrics,
    fit_multivariate_ridge,
    tail_suffix_certificate,
)
from qrc_vs_qelm.matching import (
    assert_bundles_matched,
    assert_matched,
    participation_effective_rank,
)
from qrc_vs_qelm.models import qelm
from qrc_vs_qelm.models.api import (
    QRCConfig,
    build_features,
    quantize_inputs,
    windowed_qelm_from_qrc,
)


_DISSIPATORS = {
    "local_loss": dissipators.local_loss,
    "collective_loss": dissipators.collective_loss,
    "dephasing": dissipators.dephasing,
}


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def load_profile(config_path: Path, profile_name: str) -> dict[str, Any]:
    """Load and validate one frozen exploratory profile."""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("emulation config must contain a mapping")
    if payload.get("status") != "EXPLORATORY_PILOT":
        raise ValueError("emulation config must be marked EXPLORATORY_PILOT")
    try:
        common = dict(payload["common"])
        profile = dict(payload["profiles"][profile_name])
        decision = dict(payload["pilot_decision"])
    except KeyError as exc:
        raise ValueError(f"missing emulation config key: {exc.args[0]}") from exc

    windows = tuple(_require_int("window", value, minimum=1) for value in profile["windows"])
    if windows != tuple(sorted(set(windows))):
        raise ValueError("windows must be unique and sorted")
    budgets = tuple(
        _require_int("random bank budget", value, minimum=1)
        for value in profile["random_bank_budgets"]
    )
    if budgets != tuple(sorted(set(budgets))):
        raise ValueError("random_bank_budgets must be unique and sorted")
    student_seeds = tuple(
        _require_int("student coupling seed", value, minimum=1)
        for value in profile.get("student_coupling_seeds", [])
    )
    systems = list(profile.get("systems", []))
    if not systems:
        raise ValueError("profile must contain at least one system")
    ids = [str(system["system_id"]) for system in systems]
    if len(ids) != len(set(ids)):
        raise ValueError("system_id values must be unique")
    teacher_seeds: set[int] = set()
    all_student_seeds: set[int] = set()
    for system in systems:
        system["teacher_coupling_seed"] = _require_int(
            "teacher coupling seed",
            system["teacher_coupling_seed"],
            minimum=1,
        )
        teacher_seeds.add(system["teacher_coupling_seed"])
        selected = tuple(
            _require_int("student coupling seed", value, minimum=1)
            for value in system.get("student_coupling_seeds", student_seeds)
        )
        if len(selected) < max(budgets):
            raise ValueError(
                f"student_coupling_seeds must cover the largest bank budget in {system['system_id']}"
            )
        system["student_coupling_seeds"] = selected
        all_student_seeds.update(selected)
    if teacher_seeds.intersection(all_student_seeds):
        raise ValueError("teacher and student coupling seeds must be disjoint")
    for system in systems:
        stream_seeds = []
        for field in (
            "train_input_seed",
            "val_input_seed",
            "test_input_seed",
            "pair_seed",
        ):
            system[field] = _require_int(field, system[field], minimum=1)
            stream_seeds.append(system[field])
        if len(set(stream_seeds)) != len(stream_seeds):
            raise ValueError(
                f"input and pair seeds must be disjoint within {system['system_id']}"
            )

    namespaces = {
        "teacher": teacher_seeds,
        "student": all_student_seeds,
        "train": {int(system["train_input_seed"]) for system in systems},
        "validation": {int(system["val_input_seed"]) for system in systems},
        "test": {int(system["test_input_seed"]) for system in systems},
        "pair": {int(system["pair_seed"]) for system in systems},
    }
    names = tuple(namespaces)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            if namespaces[left_name].intersection(namespaces[right_name]):
                raise ValueError(
                    f"seed namespaces {left_name} and {right_name} must be disjoint"
                )

    matched = bool(profile.get("matched_gamma_design", False))
    independent_banks = bool(profile.get("independent_student_banks", False))
    if matched:
        expected_gammas = tuple(float(value) for value in profile["matched_gammas"])
        if expected_gammas != tuple(sorted(set(expected_gammas))):
            raise ValueError("matched_gammas must be unique and sorted")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for system in systems:
            replicate_id = str(system.get("replicate_id", ""))
            if not replicate_id:
                raise ValueError("matched systems require replicate_id")
            grouped.setdefault(replicate_id, []).append(system)
        seed_fields = (
            "teacher_coupling_seed",
            "train_input_seed",
            "val_input_seed",
            "test_input_seed",
            "pair_seed",
        )
        signatures = []
        bank_signatures = []
        physical_fields = (
            "dt",
            "h_in",
            "n_qubits",
            "J_s",
            "h",
            "dissipator",
        )
        for replicate_id, rows in grouped.items():
            gammas = tuple(sorted(float(row["gamma"]) for row in rows))
            if gammas != expected_gammas:
                raise ValueError(
                    f"replicate {replicate_id} does not cover matched_gammas"
                )
            signature = tuple(int(rows[0][field]) for field in seed_fields)
            for row in rows[1:]:
                if tuple(int(row[field]) for field in seed_fields) != signature:
                    raise ValueError(
                        f"replicate {replicate_id} changes seeds across gamma"
                    )
            bank_signature = tuple(int(value) for value in rows[0]["student_coupling_seeds"])
            physical_signature = tuple(
                rows[0].get(field, common[field]) for field in physical_fields
            )
            for row in rows[1:]:
                if tuple(int(value) for value in row["student_coupling_seeds"]) != bank_signature:
                    raise ValueError(
                        f"replicate {replicate_id} changes student bank across gamma"
                    )
                if tuple(
                    row.get(field, common[field]) for field in physical_fields
                ) != physical_signature:
                    raise ValueError(
                        f"replicate {replicate_id} changes physical nuisances across gamma"
                    )
            signatures.append(signature)
            bank_signatures.append(bank_signature)
        if len(signatures) != len(set(signatures)):
            raise ValueError("matched replicate blocks must use distinct seed sets")
        if independent_banks and len(bank_signatures) != len(set(bank_signatures)):
            raise ValueError(
                "independent matched blocks must use distinct student banks"
            )
        if independent_banks:
            flattened = [seed for bank in bank_signatures for seed in bank]
            if len(flattened) != len(set(flattened)):
                raise ValueError(
                    "independent matched student banks must not share seeds"
                )

    for key in ("train_rows", "val_rows", "test_rows", "paired_histories"):
        profile[key] = _require_int(key, profile[key], minimum=1)
    for key in ("washout", "pair_burnin"):
        profile[key] = _require_int(key, profile[key], minimum=0)
    profile["pair_branch"] = _require_int(
        "pair_branch", profile["pair_branch"], minimum=1
    )
    profile["windows"] = windows
    profile["random_bank_budgets"] = budgets
    profile["student_coupling_seeds"] = student_seeds
    profile["systems"] = systems
    profile["matched_gamma_design"] = matched
    profile["independent_student_banks"] = independent_banks
    return {
        "schema_version": int(payload["schema_version"]),
        "study": str(payload["study"]),
        "status": str(payload["status"]),
        "profile_name": str(profile_name),
        "common": common,
        "profile": profile,
        "decision": decision,
    }


def _qrc_config(common: dict[str, Any], system: dict[str, Any]) -> QRCConfig:
    return QRCConfig(
        dt=float(system.get("dt", common["dt"])),
        h_in=float(system.get("h_in", common["h_in"])),
        n_virtual=int(common["n_virtual"]),
        n_qubits=int(system.get("n_qubits", common["n_qubits"])),
        J_s=float(system.get("J_s", common["J_s"])),
        h=float(system.get("h", common["h"])),
        gamma=float(system["gamma"]),
        dissipator=str(system.get("dissipator", common["dissipator"])),
        quantize_levels=int(common["quantize_levels"]),
    )


def _observables(common: dict[str, Any], n_qubits: int) -> list[readout.Observable]:
    spec = common["observables"]
    obs = readout.pauli_observables(
        int(n_qubits),
        axes=tuple(str(axis) for axis in spec["axes"]),
        max_weight=int(spec["max_weight"]),
    )
    expected = int(spec["dimension"])
    if len(obs) != expected:
        raise ValueError(
            f"observable dimension {len(obs)} does not match configured {expected}"
        )
    return obs


def _stream_inputs(
    *,
    seed: int,
    valid_prefix: int,
    rows: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.uniform(0.0, 1.0, size=int(valid_prefix) + int(rows))


def _input_blocks(
    profile: dict[str, Any],
    system: dict[str, Any],
    valid_prefix: int,
    quantize_levels: int | None,
) -> dict[str, np.ndarray]:
    raw = {
        "train": _stream_inputs(
            seed=int(system["train_input_seed"]),
            valid_prefix=valid_prefix,
            rows=int(profile["train_rows"]),
        ),
        "val": _stream_inputs(
            seed=int(system["val_input_seed"]),
            valid_prefix=valid_prefix,
            rows=int(profile["val_rows"]),
        ),
        "test": _stream_inputs(
            seed=int(system["test_input_seed"]),
            valid_prefix=valid_prefix,
            rows=int(profile["test_rows"]),
        ),
    }
    # Every model receives the same alphabet that actually drives the teacher.
    # This avoids giving the classical control unquantized inputs while the
    # teacher and quantum students receive quantized inputs.
    return {
        split: quantize_inputs(values, quantize_levels)
        for split, values in raw.items()
    }


def _bundle_blocks(
    cfg: QRCConfig | Any,
    inputs: dict[str, np.ndarray],
    coupling_seed: int,
    observables: list[readout.Observable],
    valid_prefix: int,
) -> tuple[dict[str, np.ndarray], list[Any]]:
    blocks: dict[str, np.ndarray] = {}
    bundles = []
    for split in ("train", "val", "test"):
        bundle = build_features(
            cfg,
            inputs[split],
            int(coupling_seed),
            observables=observables,
        )
        if bundle.first_valid > int(valid_prefix):
            raise RuntimeError("shared valid prefix does not cover model warm-up")
        blocks[split] = bundle.features[int(valid_prefix) :].copy()
        bundles.append(bundle)
    return blocks, bundles


def _delay_blocks(
    inputs: dict[str, np.ndarray],
    window: int,
    valid_prefix: int,
) -> dict[str, np.ndarray]:
    return {
        split: qelm.delay_embed(values, int(window))[int(valid_prefix) :].copy()
        for split, values in inputs.items()
    }


def _stack_blocks(
    blocks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(blocks["train"], dtype=float)
    val = np.asarray(blocks["val"], dtype=float)
    test = np.asarray(blocks["test"], dtype=float)
    values = np.concatenate([train, val, test], axis=0)
    train_idx = np.arange(0, len(train), dtype=int)
    val_idx = np.arange(len(train), len(train) + len(val), dtype=int)
    test_idx = np.arange(len(train) + len(val), len(values), dtype=int)
    return values, train_idx, val_idx, test_idx


def _fit_student(
    feature_blocks: dict[str, np.ndarray],
    target_blocks: dict[str, np.ndarray],
    alpha_grid: tuple[float, ...],
) -> tuple[MultiOutputRidge, dict[str, Any]]:
    features, train_idx, val_idx, test_idx = _stack_blocks(feature_blocks)
    targets, y_train, y_val, y_test = _stack_blocks(target_blocks)
    if not (
        np.array_equal(train_idx, y_train)
        and np.array_equal(val_idx, y_val)
        and np.array_equal(test_idx, y_test)
    ):
        raise RuntimeError("feature and target split rows diverged")
    model = fit_multivariate_ridge(
        features,
        targets,
        train_idx,
        val_idx,
        alphas=alpha_grid,
    )
    prediction = model.predict(features[test_idx])
    metrics = emulation_metrics(targets[test_idx], prediction)
    return model, {
        **asdict(metrics),
        "alpha": model.alpha,
        "validation_nrmse": model.validation_nrmse,
        "feature_dim": int(features.shape[1]),
        "effective_rank": float(participation_effective_rank(features[train_idx])),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
    }


def _direct_metrics(
    prediction_blocks: dict[str, np.ndarray],
    target_blocks: dict[str, np.ndarray],
) -> dict[str, Any]:
    metrics = emulation_metrics(target_blocks["test"], prediction_blocks["test"])
    return {
        **asdict(metrics),
        "alpha": None,
        "validation_nrmse": None,
        "feature_dim": int(prediction_blocks["test"].shape[1]),
        "effective_rank": float(
            participation_effective_rank(prediction_blocks["train"])
        ),
        "n_train": int(len(prediction_blocks["train"])),
        "n_val": int(len(prediction_blocks["val"])),
        "n_test": int(len(prediction_blocks["test"])),
    }


def _constant_metrics(target_blocks: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    mean = target_blocks["train"].mean(axis=0)
    prediction = np.repeat(mean[None, :], len(target_blocks["test"]), axis=0)
    metrics = emulation_metrics(target_blocks["test"], prediction)
    return mean, {
        **asdict(metrics),
        "alpha": None,
        "validation_nrmse": None,
        "feature_dim": 0,
        "effective_rank": 0.0,
        "n_train": int(len(target_blocks["train"])),
        "n_val": int(len(target_blocks["val"])),
        "n_test": int(len(target_blocks["test"])),
    }


def _reservoir(
    cfg: QRCConfig,
    coupling_seed: int,
) -> reservoirs.LindbladReservoir:
    rng = np.random.default_rng(int(coupling_seed))
    couplings = reservoirs.random_couplings(cfg.n_qubits, cfg.J_s, rng)
    h_static = reservoirs.ising_xx_hamiltonian(couplings, cfg.h, cfg.n_qubits)
    h_drive = cfg.h_in * reservoirs.transverse_drive(cfg.n_qubits)
    if cfg.gamma == 0.0:
        jumps = []
    else:
        jumps = _DISSIPATORS[cfg.dissipator](cfg.n_qubits, cfg.gamma)
    return reservoirs.LindbladReservoir.from_terms(
        cfg.n_qubits,
        h_static,
        h_drive,
        jumps,
        cfg.dt,
        cache_propagators=cfg.quantize_levels is not None,
        quantize=cfg.quantize_levels,
    )


def _feature(state: np.ndarray, observables: list[readout.Observable]) -> np.ndarray:
    return readout.features_from_states([state], observables).reshape(-1)


def _evolve(
    reservoir: reservoirs.LindbladReservoir,
    state: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    current = state.copy()
    for value in np.asarray(inputs, dtype=float):
        current = reservoir.step(current, float(value))
    return current


def _production_replay_feature(
    reservoir: reservoirs.LindbladReservoir,
    history: np.ndarray,
    window: int,
    observables: list[readout.Observable],
) -> np.ndarray:
    # Exercise the production delay embedding separately on each complete
    # history arm. Equality is audited instead of forced by copying one suffix
    # prediction into both arms.
    newest_first = qelm.delay_embed(np.asarray(history, dtype=float), int(window))[-1]
    states = qelm.time_delayed_window_states(
        reservoir,
        newest_first[::-1],
        n_snapshots=1,
    )
    return readout.features_from_states(states, observables).reshape(-1)


def _paired_panel(
    *,
    cfg: QRCConfig,
    observables: list[readout.Observable],
    windows: tuple[int, ...],
    profile: dict[str, Any],
    system: dict[str, Any],
    trained: dict[tuple[str, int], MultiOutputRidge | None],
    train_mean: np.ndarray,
    student_seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    teacher = _reservoir(cfg, int(system["teacher_coupling_seed"]))
    students = [_reservoir(cfg, seed) for seed in student_seeds]
    endpoints = (0, *windows)
    max_window = max(windows)
    teacher_a: dict[int, list[np.ndarray]] = {value: [] for value in endpoints}
    teacher_b: dict[int, list[np.ndarray]] = {value: [] for value in endpoints}
    trace_distances: dict[int, list[float]] = {value: [] for value in endpoints}
    predictions_a: dict[tuple[str, int, int], list[np.ndarray]] = {}
    predictions_b: dict[tuple[str, int, int], list[np.ndarray]] = {}

    model_names_by_window: dict[int, tuple[str, ...]] = {}
    for window in windows:
        names = ["oracle_identity", "oracle_fitted", "linear_delay", "constant"]
        names.extend(f"random_bank_R{budget}" for budget in budgets)
        model_names_by_window[window] = tuple(names)
        for endpoint in endpoints:
            if endpoint < window:
                continue
            for name in names:
                predictions_a[(name, window, endpoint)] = []
                predictions_b[(name, window, endpoint)] = []

    for pair_id in range(int(profile["paired_histories"])):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(system["pair_seed"]), pair_id])
        )
        common = quantize_inputs(
            rng.uniform(0.0, 1.0, size=int(profile["pair_burnin"])),
            cfg.quantize_levels,
        )
        branch_a = quantize_inputs(
            rng.uniform(0.0, 1.0, size=int(profile["pair_branch"])),
            cfg.quantize_levels,
        )
        branch_b = quantize_inputs(
            rng.uniform(0.0, 1.0, size=int(profile["pair_branch"])),
            cfg.quantize_levels,
        )
        suffix = quantize_inputs(
            rng.uniform(0.0, 1.0, size=max_window), cfg.quantize_levels
        )
        base_history_a = np.concatenate([common, branch_a])
        base_history_b = np.concatenate([common, branch_b])

        initial = teacher.initial_state()
        common_state = _evolve(teacher, initial, common)
        carry_a = _evolve(teacher, common_state, branch_a)
        carry_b = _evolve(teacher, common_state, branch_b)
        for length in range(max_window + 1):
            if length in teacher_a:
                teacher_a[length].append(_feature(carry_a, observables))
                teacher_b[length].append(_feature(carry_b, observables))
                trace_distances[length].append(
                    diagnostics.trace_distance(carry_a, carry_b)
                )
            if length == max_window:
                break
            value = float(suffix[length])
            carry_a = teacher.step(carry_a, value)
            carry_b = teacher.step(carry_b, value)

        for window in windows:
            for endpoint in endpoints:
                if endpoint < window:
                    continue
                history_a = np.concatenate([base_history_a, suffix[:endpoint]])
                history_b = np.concatenate([base_history_b, suffix[:endpoint]])
                oracle_a = _production_replay_feature(
                    teacher, history_a, window, observables
                )
                oracle_b = _production_replay_feature(
                    teacher, history_b, window, observables
                )
                student_vectors_a = [
                    _production_replay_feature(
                        student, history_a, window, observables
                    )
                    for student in students
                ]
                student_vectors_b = [
                    _production_replay_feature(
                        student, history_b, window, observables
                    )
                    for student in students
                ]
                feature_vectors_a: dict[str, np.ndarray] = {
                    "oracle_identity": oracle_a,
                    "oracle_fitted": oracle_a,
                    "linear_delay": qelm.delay_embed(history_a, int(window))[-1],
                    "constant": np.empty(0, dtype=float),
                }
                feature_vectors_b: dict[str, np.ndarray] = {
                    "oracle_identity": oracle_b,
                    "oracle_fitted": oracle_b,
                    "linear_delay": qelm.delay_embed(history_b, int(window))[-1],
                    "constant": np.empty(0, dtype=float),
                }
                for budget in budgets:
                    feature_vectors_a[f"random_bank_R{budget}"] = np.concatenate(
                        student_vectors_a[:budget]
                    )
                    feature_vectors_b[f"random_bank_R{budget}"] = np.concatenate(
                        student_vectors_b[:budget]
                    )
                for name in model_names_by_window[window]:
                    if name == "oracle_identity":
                        prediction_a = oracle_a
                        prediction_b = oracle_b
                    elif name == "constant":
                        prediction_a = train_mean
                        prediction_b = train_mean
                    else:
                        model = trained[(name, window)]
                        if model is None:
                            raise RuntimeError(f"missing trained model {name}, d={window}")
                        prediction_a = model.predict(
                            feature_vectors_a[name][None, :]
                        )[0]
                        prediction_b = model.predict(
                            feature_vectors_b[name][None, :]
                        )[0]
                    predictions_a[(name, window, endpoint)].append(
                        prediction_a.copy()
                    )
                    predictions_b[(name, window, endpoint)].append(
                        prediction_b.copy()
                    )

    trace_rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        values = np.asarray(trace_distances[endpoint], dtype=float)
        trace_rows.append(
            {
                "system_id": str(system["system_id"]),
                "replicate_id": str(system.get("replicate_id", system["system_id"])),
                "gamma": cfg.gamma,
                "suffix_endpoint": endpoint,
                "trace_distance_mean": float(np.mean(values)),
                "trace_distance_max": float(np.max(values)),
            }
        )

    certificate_rows: list[dict[str, Any]] = []
    d0_first = np.stack([np.asarray(teacher_a[value]) for value in endpoints])
    d0_second = np.stack([np.asarray(teacher_b[value]) for value in endpoints])
    d0_zero = np.zeros_like(d0_first)
    d0_certificate = tail_suffix_certificate(
        endpoints, d0_first, d0_second, d0_zero, atol=tolerance
    )
    baseline = {
        "Dinf_d0": d0_certificate.suffix_diameter,
        "D2_d0": d0_certificate.rms_suffix_separation,
    }

    for window in windows:
        tail_endpoints = tuple(value for value in endpoints if value >= window)
        outputs_a = np.stack(
            [np.asarray(teacher_a[value]) for value in tail_endpoints]
        )
        outputs_b = np.stack(
            [np.asarray(teacher_b[value]) for value in tail_endpoints]
        )
        for name in model_names_by_window[window]:
            pred_a = np.stack(
                [
                    np.asarray(predictions_a[(name, window, value)])
                    for value in tail_endpoints
                ]
            )
            pred_b = np.stack(
                [
                    np.asarray(predictions_b[(name, window, value)])
                    for value in tail_endpoints
                ]
            )
            certificate = tail_suffix_certificate(
                tail_endpoints,
                outputs_a,
                outputs_b,
                pred_a,
                pred_b,
                atol=tolerance,
            )
            certificate_rows.append(
                {
                    "system_id": str(system["system_id"]),
                    "replicate_id": str(system.get("replicate_id", system["system_id"])),
                    "gamma": cfg.gamma,
                    "model": name,
                    "window": window,
                    **asdict(certificate),
                    "Dinf_fraction_of_d0": (
                        certificate.suffix_diameter / baseline["Dinf_d0"]
                        if baseline["Dinf_d0"] > 0
                        else float("nan")
                    ),
                    "D2_fraction_of_d0": (
                        certificate.rms_suffix_separation / baseline["D2_d0"]
                        if baseline["D2_d0"] > 0
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(certificate_rows), pd.DataFrame(trace_rows), baseline


def _stream_row(
    *,
    system: dict[str, Any],
    cfg: QRCConfig,
    model: str,
    window: int,
    budget: int,
    metrics: dict[str, Any],
    output_dim: int,
) -> dict[str, Any]:
    feature_dim = int(metrics["feature_dim"])
    quantum_model = model.startswith("oracle") or model.startswith("random_bank")
    updates = int(window * budget) if quantum_model else 0
    expectations = int(output_dim * budget) if quantum_model else 0
    return {
        "system_id": str(system["system_id"]),
        "replicate_id": str(system.get("replicate_id", system["system_id"])),
        "gamma": cfg.gamma,
        "dissipator": cfg.dissipator,
        "model": model,
        "window": int(window),
        "bank_budget": int(budget),
        "quantum_updates_per_prediction": updates,
        "expectations_per_prediction": expectations,
        "readout_parameters": (
            0 if model == "oracle_identity" else int(output_dim * (feature_dim + 1))
        ),
        **metrics,
    }


def run_system(
    *,
    study: dict[str, Any],
    system: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    common = study["common"]
    profile = study["profile"]
    windows = tuple(profile["windows"])
    budgets = tuple(profile["random_bank_budgets"])
    student_seeds = tuple(
        int(value)
        for value in system.get(
            "student_coupling_seeds",
            profile["student_coupling_seeds"],
        )
    )
    cfg = _qrc_config(common, system)
    observables = _observables(common, cfg.n_qubits)
    max_window = max(windows)
    valid_prefix = max(int(profile["washout"]), max_window - 1)
    inputs = _input_blocks(
        profile,
        system,
        valid_prefix,
        cfg.quantize_levels,
    )
    target_blocks, target_bundles = _bundle_blocks(
        cfg,
        inputs,
        int(system["teacher_coupling_seed"]),
        observables,
        valid_prefix,
    )
    alpha_grid = tuple(
        float(value)
        for value in profile.get("alpha_grid", common["alpha_grid"])
    )
    output_dim = len(observables)
    stream_rows: list[dict[str, Any]] = []
    trained: dict[tuple[str, int], MultiOutputRidge | None] = {}
    train_mean, constant = _constant_metrics(target_blocks)

    for window in windows:
        qelm_cfg = windowed_qelm_from_qrc(cfg, window_k=window)
        assert_matched(cfg, qelm_cfg, observables=observables)
        oracle_blocks, oracle_bundles = _bundle_blocks(
            qelm_cfg,
            inputs,
            int(system["teacher_coupling_seed"]),
            observables,
            valid_prefix,
        )
        for teacher_bundle, oracle_bundle in zip(target_bundles, oracle_bundles):
            assert_bundles_matched(teacher_bundle, oracle_bundle)

        oracle_direct = _direct_metrics(oracle_blocks, target_blocks)
        stream_rows.append(
            _stream_row(
                system=system,
                cfg=cfg,
                model="oracle_identity",
                window=window,
                budget=1,
                metrics=oracle_direct,
                output_dim=output_dim,
            )
        )
        oracle_model, oracle_fit = _fit_student(
            oracle_blocks, target_blocks, alpha_grid
        )
        trained[("oracle_fitted", window)] = oracle_model
        stream_rows.append(
            _stream_row(
                system=system,
                cfg=cfg,
                model="oracle_fitted",
                window=window,
                budget=1,
                metrics=oracle_fit,
                output_dim=output_dim,
            )
        )

        bank_blocks: list[dict[str, np.ndarray]] = []
        for student_seed in student_seeds:
            blocks, bundles = _bundle_blocks(
                qelm_cfg,
                inputs,
                student_seed,
                observables,
                valid_prefix,
            )
            for teacher_bundle, student_bundle in zip(target_bundles, bundles):
                assert_bundles_matched(teacher_bundle, student_bundle)
            bank_blocks.append(blocks)
        for budget in budgets:
            combined = {
                split: np.concatenate(
                    [bank_blocks[idx][split] for idx in range(budget)], axis=1
                )
                for split in ("train", "val", "test")
            }
            name = f"random_bank_R{budget}"
            model, metrics = _fit_student(combined, target_blocks, alpha_grid)
            trained[(name, window)] = model
            stream_rows.append(
                _stream_row(
                    system=system,
                    cfg=cfg,
                    model=name,
                    window=window,
                    budget=budget,
                    metrics=metrics,
                    output_dim=output_dim,
                )
            )

        delay = _delay_blocks(inputs, window, valid_prefix)
        delay_model, delay_metrics = _fit_student(delay, target_blocks, alpha_grid)
        trained[("linear_delay", window)] = delay_model
        stream_rows.append(
            _stream_row(
                system=system,
                cfg=cfg,
                model="linear_delay",
                window=window,
                budget=0,
                metrics=delay_metrics,
                output_dim=output_dim,
            )
        )
        trained[("constant", window)] = None
        trained[("oracle_identity", window)] = None
        stream_rows.append(
            _stream_row(
                system=system,
                cfg=cfg,
                model="constant",
                window=window,
                budget=0,
                metrics=constant,
                output_dim=output_dim,
            )
        )

    tolerance = float(study["decision"]["numeric_tolerance"])
    certificates, trace, baseline = _paired_panel(
        cfg=cfg,
        observables=observables,
        windows=windows,
        profile=profile,
        system=system,
        trained=trained,
        train_mean=train_mean,
        student_seeds=student_seeds,
        budgets=budgets,
        tolerance=tolerance,
    )

    reservoir = _reservoir(cfg, int(system["teacher_coupling_seed"]))
    gap = float(diagnostics.spectral_gap(reservoir.liouvillian(0.5)))
    mixing_steps = float("inf") if gap <= 0.0 else 1.0 / (gap * cfg.dt)
    target_variance = float(np.mean(np.var(target_blocks["test"], axis=0)))
    system_row = {
        "system_id": str(system["system_id"]),
        "replicate_id": str(system.get("replicate_id", system["system_id"])),
        "gamma": cfg.gamma,
        "dissipator": cfg.dissipator,
        "teacher_coupling_seed": int(system["teacher_coupling_seed"]),
        "student_coupling_seeds": "|".join(str(value) for value in student_seeds),
        "spectral_gap_at_s_0_5": gap,
        "mixing_steps_proxy": mixing_steps,
        "teacher_test_variance": target_variance,
        **baseline,
    }
    return pd.DataFrame(stream_rows), certificates, trace, system_row


def evaluate_decision(
    stream: pd.DataFrame,
    certificates: pd.DataFrame,
    systems: pd.DataFrame,
    trace: pd.DataFrame,
    decision: dict[str, Any],
    *,
    matched_gamma_design: bool,
) -> dict[str, Any]:
    """Apply the frozen exploratory pilot decision rules."""

    tolerance = float(decision["numeric_tolerance"])
    e0 = bool(
        certificates["bound_satisfied"].astype(bool).all()
        and (certificates["prediction_disagreement"] <= tolerance).all()
    )
    live = systems["D2_d0"] >= float(decision["liveness_D2_at_d0"])
    live_fraction = float(live.mean())
    e1 = live_fraction >= float(decision["min_live_system_fraction"])

    min_window = int(stream["window"].min())
    max_window = int(stream["window"].max())
    dissipative_ids = systems.loc[systems["gamma"] > 0.0, "system_id"].tolist()

    # The certificate panel and the stream-test panel use independent inputs.
    # Compute the association within each system to avoid a pooled-scale
    # correlation that is mathematically or between-system induced.
    oracle_certificate = certificates.loc[
        certificates["model"] == "oracle_identity",
        ["system_id", "window", "rms_suffix_separation"],
    ]
    oracle_stream = stream.loc[
        stream["model"] == "oracle_identity",
        ["system_id", "window", "nrmse"],
    ]
    association_frame = oracle_certificate.merge(
        oracle_stream,
        on=["system_id", "window"],
        validate="one_to_one",
    )
    associations: dict[str, float | None] = {}
    for system_id, group in association_frame.groupby("system_id"):
        statistic = float(
            spearmanr(group["rms_suffix_separation"], group["nrmse"]).statistic
        ) if len(group) >= 3 else float("nan")
        associations[str(system_id)] = statistic if np.isfinite(statistic) else None
    association_pass = [
        associations.get(str(system_id)) is not None
        and float(associations[str(system_id)])
        >= float(decision["min_oracle_error_certificate_spearman"])
        for system_id in dissipative_ids
    ]
    association_fraction = (
        float(np.mean(association_pass)) if association_pass else 0.0
    )
    e2 = association_fraction >= float(
        decision["min_associated_dissipative_system_fraction"]
    )

    oracle = stream.loc[stream["model"] == "oracle_identity"]
    reductions: dict[str, float] = {}
    for system_id, group in oracle.groupby("system_id"):
        small = float(group.loc[group["window"] == min_window, "nrmse"].iloc[0])
        large = float(group.loc[group["window"] == max_window, "nrmse"].iloc[0])
        reductions[str(system_id)] = 1.0 - large / max(small, 1e-30)
    threshold = float(decision["min_oracle_max_window_reduction"])
    improved_fraction = float(
        np.mean([reductions[str(system_id)] >= threshold for system_id in dissipative_ids])
    ) if dissipative_ids else 0.0
    e3 = improved_fraction >= float(
        decision["min_dissipative_system_fraction_improved"]
    )

    budget = int(decision["generic_student_budget"])
    generic_name = f"random_bank_R{budget}"
    generic = stream.loc[stream["model"] == generic_name]
    generic_pass: dict[str, bool] = {}
    generic_reductions: dict[str, float] = {}
    for system_id, group in generic.groupby("system_id"):
        small = float(group.loc[group["window"] == min_window, "nrmse"].iloc[0])
        large = float(group.loc[group["window"] == max_window, "nrmse"].iloc[0])
        reduction = 1.0 - large / max(small, 1e-30)
        generic_reductions[str(system_id)] = reduction
        generic_pass[str(system_id)] = bool(
            large <= float(decision["generic_student_max_window_nrmse"])
            or reduction >= float(decision["generic_student_min_reduction"])
        )
    generic_dissipative = [
        generic_pass.get(str(system_id), False) for system_id in dissipative_ids
    ]
    generic_fraction = float(np.mean(generic_dissipative)) if generic_dissipative else 0.0
    e4_window = generic_fraction >= float(
        decision["min_dissipative_system_fraction_improved"]
    )

    # Explicitly test the second resource axis: does the largest nested bank
    # improve over R=1 at the same largest window?
    feature_gains: dict[str, float] = {}
    smallest_name = "random_bank_R1"
    smallest = stream.loc[
        (stream["model"] == smallest_name) & (stream["window"] == max_window)
    ]
    largest = generic.loc[generic["window"] == max_window]
    for system_id in systems["system_id"]:
        first = smallest.loc[smallest["system_id"] == system_id, "nrmse"]
        last = largest.loc[largest["system_id"] == system_id, "nrmse"]
        if not first.empty and not last.empty:
            feature_gains[str(system_id)] = 1.0 - float(last.iloc[0]) / max(
                float(first.iloc[0]), 1e-30
            )
    feature_pass = [
        feature_gains.get(str(system_id), -float("inf"))
        >= float(decision["min_feature_budget_reduction"])
        for system_id in dissipative_ids
    ]
    feature_fraction = float(np.mean(feature_pass)) if feature_pass else 0.0
    e4_feature = feature_fraction >= float(
        decision["min_feature_budget_system_fraction"]
    )

    # A QELM-bank interpretation also requires that R=max beat the matched
    # linear-delay baseline by a frozen margin at d=max.
    linear = stream.loc[
        (stream["model"] == "linear_delay") & (stream["window"] == max_window)
    ]
    linear_gains: dict[str, float] = {}
    for system_id in systems["system_id"]:
        quantum = largest.loc[largest["system_id"] == system_id, "nrmse"]
        classical = linear.loc[linear["system_id"] == system_id, "nrmse"]
        if not quantum.empty and not classical.empty:
            linear_gains[str(system_id)] = 1.0 - float(quantum.iloc[0]) / max(
                float(classical.iloc[0]), 1e-30
            )
    linear_pass = [
        linear_gains.get(str(system_id), -float("inf"))
        >= float(decision["min_generic_gain_over_linear"])
        for system_id in dissipative_ids
    ]
    linear_fraction = float(np.mean(linear_pass)) if linear_pass else 0.0
    e4_control = linear_fraction >= float(
        decision["min_generic_over_linear_system_fraction"]
    )

    # Use state trace distance—not an observable projection—for the non-fading
    # unitary boundary.
    unitary_ids = systems.loc[systems["gamma"] == 0.0, "system_id"].tolist()
    trace_remaining: dict[str, float] = {}
    for system_id in unitary_ids:
        group = trace.loc[trace["system_id"] == system_id]
        initial = group.loc[group["suffix_endpoint"] == 0, "trace_distance_mean"]
        final = group.loc[
            group["suffix_endpoint"] == max_window, "trace_distance_mean"
        ]
        if not initial.empty and not final.empty:
            trace_remaining[str(system_id)] = float(final.iloc[0]) / max(
                float(initial.iloc[0]), 1e-30
            )
    nonfading_pass = [
        trace_remaining.get(str(system_id), -float("inf"))
        >= float(decision["nonfading_min_remaining_trace_fraction"])
        for system_id in unitary_ids
    ]
    nonfading_fraction = (
        float(np.mean(nonfading_pass)) if nonfading_pass else 0.0
    )
    e5 = nonfading_fraction >= float(
        decision["min_nonfading_system_fraction"]
    )

    # In each matched block, stronger dissipation should leave a smaller
    # normalized suffix-memory witness at d=max.
    matched_associations: dict[str, float | None] = {}
    matched_fraction: float | None = None
    e6: bool | None = None
    if matched_gamma_design:
        endpoint_memory = certificates.loc[
            (certificates["model"] == "oracle_identity")
            & (certificates["window"] == max_window),
            ["system_id", "D2_fraction_of_d0"],
        ].merge(
            systems[["system_id", "replicate_id", "gamma"]],
            on="system_id",
            validate="one_to_one",
        )
        matched_pass = []
        for replicate_id, group in endpoint_memory.groupby("replicate_id"):
            statistic = float(
                spearmanr(group["gamma"], group["D2_fraction_of_d0"]).statistic
            ) if len(group) >= 3 else float("nan")
            matched_associations[str(replicate_id)] = (
                statistic if np.isfinite(statistic) else None
            )
            if np.isfinite(statistic):
                matched_pass.append(
                    statistic <= float(decision["max_matched_gamma_memory_spearman"])
                )
        matched_fraction = float(np.mean(matched_pass)) if matched_pass else 0.0
        e6 = matched_fraction >= float(
            decision["min_matched_replicate_fraction_ordered"]
        )

    e6_ok = True if e6 is None else e6

    if not e0:
        verdict = "NO_GO_IMPLEMENTATION"
    elif e1 and e2 and e3 and e5 and e6_ok and e4_window and e4_feature and e4_control:
        verdict = "PILOT_SUPPORTS_TWO_RESOURCE_SCALING"
    elif e1 and e2 and e3 and e5 and e6_ok:
        verdict = "PILOT_SUPPORTS_CONSTRUCTIVE_TRUNCATION"
    else:
        verdict = "PILOT_NO_GO"
    return {
        "verdict": verdict,
        "E0_suffix_certificate_implementation": e0,
        "E1_liveness": e1,
        "E2_memory_error_association": e2,
        "E3_constructive_convergence": e3,
        "E4a_generic_window_emulation": e4_window,
        "E4b_feature_budget_gain": e4_feature,
        "E4c_generic_beats_linear": e4_control,
        "E5_nonfading_boundary": e5,
        "E6_matched_mixing_order": e6,
        "live_system_fraction": live_fraction,
        "oracle_certificate_spearman_by_system": associations,
        "associated_dissipative_system_fraction": association_fraction,
        "oracle_reduction_by_system": reductions,
        "oracle_dissipative_improved_fraction": improved_fraction,
        "generic_reduction_by_system": generic_reductions,
        "generic_dissipative_pass_fraction": generic_fraction,
        "feature_budget_gain_by_system": feature_gains,
        "feature_budget_pass_fraction": feature_fraction,
        "generic_gain_over_linear_by_system": linear_gains,
        "generic_over_linear_pass_fraction": linear_fraction,
        "unitary_remaining_trace_fraction_by_system": trace_remaining,
        "unitary_nonfading_pass_fraction": nonfading_fraction,
        "matched_gamma_memory_spearman_by_replicate": matched_associations,
        "matched_replicate_pass_fraction": matched_fraction,
        "min_window": min_window,
        "max_window": max_window,
    }


def _panel_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Use gamma-level replicate summaries when matched replicates are present."""

    if (
        "replicate_id" in frame
        and frame["system_id"].nunique() > frame["gamma"].nunique()
    ):
        return [
            (f"$\\gamma={float(gamma):g}$", group.copy())
            for gamma, group in frame.groupby("gamma", sort=True)
        ]
    return [
        (str(system_id), frame.loc[frame["system_id"] == system_id].copy())
        for system_id in dict.fromkeys(frame["system_id"].tolist())
    ]


def _plot_stream_curves(stream: pd.DataFrame, out_path: Path) -> None:
    panels = _panel_groups(stream)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
    available = set(stream["model"])
    bank_models = sorted(
        (name for name in available if name.startswith("random_bank_R")),
        key=lambda name: int(name.rsplit("R", 1)[1]),
    )
    selected = ["oracle_identity"]
    if bank_models:
        selected.append(bank_models[0])
        if bank_models[-1] != bank_models[0]:
            selected.append(bank_models[-1])
    selected.append("linear_delay")
    selected = [name for name in selected if name in available]
    styles = {
        "oracle_identity": ("black", "o"),
        "linear_delay": ("#7570b3", "D"),
    }
    if bank_models:
        styles[bank_models[0]] = ("#d95f02", "s")
        styles[bank_models[-1]] = ("#1b9e77", "^")
    for axis, (title, group) in zip(axes[0], panels):
        for model in selected:
            rows = group.loc[group["model"] == model]
            curve = rows.groupby("window")["nrmse"].agg(
                median="median",
                lower=lambda values: values.quantile(0.25),
                upper=lambda values: values.quantile(0.75),
            ).reset_index()
            color, marker = styles[model]
            axis.plot(
                curve["window"],
                curve["median"],
                marker=marker,
                color=color,
                label=model,
            )
            if rows["system_id"].nunique() > 1:
                axis.fill_between(
                    curve["window"],
                    curve["lower"],
                    curve["upper"],
                    color=color,
                    alpha=0.15,
                )
        axis.set_title(title)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("window d")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("test observable NRMSE")
    axes[0, -1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_certificates(certificates: pd.DataFrame, out_path: Path) -> None:
    oracle = certificates.loc[certificates["model"] == "oracle_identity"]
    panels = _panel_groups(oracle)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
    for axis, (title, group) in zip(axes[0], panels):
        for column, label, color, marker in (
            ("paired_rmse_lower_bound", "$D_2(d)/2$ certificate", "#7570b3", "o"),
            ("paired_rmse_at_witness", "oracle paired RMSE", "#d95f02", "s"),
        ):
            curve = group.groupby("window")[column].agg(
                median="median",
                lower=lambda values: values.quantile(0.25),
                upper=lambda values: values.quantile(0.75),
            ).reset_index()
            axis.plot(
                curve["window"],
                curve["median"],
                marker=marker,
                label=label,
                color=color,
            )
            if group["system_id"].nunique() > 1:
                axis.fill_between(
                    curve["window"],
                    curve["lower"],
                    curve["upper"],
                    color=color,
                    alpha=0.15,
                )
        axis.set_title(title)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("window d")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("observable RMS error")
    axes[0, -1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_resource_surface(stream: pd.DataFrame, out_path: Path) -> None:
    random_rows = stream.loc[stream["model"].str.startswith("random_bank_R")]
    panels = _panel_groups(random_rows)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), squeeze=False)
    log_values = np.log10(np.maximum(random_rows["nrmse"].to_numpy(), 1e-12))
    color_min = float(np.min(log_values))
    color_max = float(np.max(log_values))
    for axis, (title, group) in zip(axes[0], panels):
        table = group.pivot_table(
            index="bank_budget",
            columns="window",
            values="nrmse",
            aggfunc="median",
        ).sort_index()
        image = axis.imshow(
            np.log10(np.maximum(table.to_numpy(), 1e-12)),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
        )
        axis.set_xticks(range(len(table.columns)), [str(value) for value in table.columns], rotation=45)
        axis.set_yticks(range(len(table.index)), [str(value) for value in table.index])
        axis.set_xlabel("window d")
        axis.set_ylabel("QELM bank R")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.72, label="log10 test NRMSE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_study(
    *,
    config_path: Path,
    profile_name: str,
    out_dir: Path,
) -> dict[str, Any]:
    study = load_profile(config_path, profile_name)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    stream_frames = []
    certificate_frames = []
    trace_frames = []
    system_rows = []
    for system in study["profile"]["systems"]:
        stream, certificates, trace, system_row = run_system(
            study=study,
            system=system,
        )
        stream_frames.append(stream)
        certificate_frames.append(certificates)
        trace_frames.append(trace)
        system_rows.append(system_row)

    stream = pd.concat(stream_frames, ignore_index=True)
    certificates = pd.concat(certificate_frames, ignore_index=True)
    trace = pd.concat(trace_frames, ignore_index=True)
    systems = pd.DataFrame(system_rows)
    decision = evaluate_decision(
        stream,
        certificates,
        systems,
        trace,
        study["decision"],
        matched_gamma_design=bool(
            study["profile"].get("matched_gamma_design", False)
        ),
    )

    paths = {
        "stream": out_dir / "stream_emulation_metrics.csv",
        "certificates": out_dir / "paired_suffix_certificates.csv",
        "trace": out_dir / "trace_distance_decay.csv",
        "systems": out_dir / "system_diagnostics.csv",
        "summary": out_dir / "emulation_pilot_summary.json",
        "stream_figure": out_dir / "emulation_curves.png",
        "certificate_figure": out_dir / "certificate_sandwich.png",
        "resource_figure": out_dir / "resource_surface.png",
    }
    for frame, key in (
        (stream, "stream"),
        (certificates, "certificates"),
        (trace, "trace"),
        (systems, "systems"),
    ):
        frame.to_csv(paths[key], index=False, float_format="%.17g")
    summary = {
        "study": study["study"],
        "status": study["status"],
        "profile": profile_name,
        "decision": decision,
        "counts": {
            "systems": int(len(systems)),
            "stream_metric_rows": int(len(stream)),
            "certificate_rows": int(len(certificates)),
            "paired_histories_per_system": int(study["profile"]["paired_histories"]),
        },
        "scope": {
            "exact_simulator_expectations_only": True,
            "teacher_target": "full accessible QRC Pauli vector",
            "confirmatory": False,
            "held_out_bank_seed_replication": bool(
                study["profile"].get("independent_student_banks", False)
            ),
        },
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot_stream_curves(stream, paths["stream_figure"])
    _plot_certificates(certificates, paths["certificate_figure"])
    _plot_resource_surface(stream, paths["resource_figure"])

    manifests.write_manifest(
        out_dir,
        "emulation_pilot",
        {
            "repo_root": ROOT,
            "study": study["study"],
            "profile": profile_name,
            "exploratory": True,
            "config_path": config_path.resolve(),
            "config_sha256": manifests.sha256_file(config_path),
            "runner_sha256": manifests.sha256_file(Path(__file__)),
            "emulation_module_sha256": manifests.sha256_file(
                ROOT / "src" / "qrc_vs_qelm" / "emulation.py"
            ),
            "frame_sha256": {
                "stream": manifests.sha256_frame(stream),
                "certificates": manifests.sha256_frame(certificates),
                "trace": manifests.sha256_frame(trace),
                "systems": manifests.sha256_frame(systems),
            },
            "decision": decision,
        },
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "emulation_pilot.yaml",
    )
    parser.add_argument(
        "--profile",
        choices=(
            "smoke",
            "pilot",
            "matched_pilot",
            "replicated_pilot",
            "replicated_pilot_wide_ridge",
            "replicated_pilot_intercept_complete",
        ),
        default="smoke",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out or ROOT / f"results_emulation_{args.profile}"
    summary = run_study(
        config_path=args.config.resolve(),
        profile_name=str(args.profile),
        out_dir=out_dir.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
