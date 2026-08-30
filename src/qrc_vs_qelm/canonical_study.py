"""Runner core for the canonical QRC-to-parallel-QELM emulation study."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from qrc_vs_qelm.canonical_emulation import (
    classify_emulation_regime,
    degree_two_delay_features,
    fit_random_tanh_feature_map,
    select_nested_snapshots,
    suffix_risk_decomposition,
    target_projection_diagnostic,
)
from qrc_vs_qelm.canonical_resources import (
    canonical_resource_report,
    observable_names_sha256,
)
from qrc_vs_qelm.core import dissipators, readout, reservoirs
from qrc_vs_qelm.emulation import (
    MultiOutputRidge,
    RIDGE_LSTSQ_RCOND,
    emulation_metrics,
    fit_multivariate_ridge,
)
from qrc_vs_qelm.matching import participation_effective_rank
from qrc_vs_qelm.models import qelm
from qrc_vs_qelm.provenance import (
    canonical_reproducibility_snapshot,
    git_worktree_manifest,
    verify_canonical_freeze_record,
)


_DISSIPATORS = {
    "local_loss": dissipators.local_loss,
    "collective_loss": dissipators.collective_loss,
    "dephasing": dissipators.dephasing,
}
_SEED_FIELDS = (
    "student_coupling_seed",
    "train_input_seed",
    "val_input_seed",
    "test_input_seed",
    "suffix_input_seed",
    "prefix_input_seed",
    "classical_feature_seed",
)


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    checked = int(value)
    if checked < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return checked


def _float_grid(values: Iterable[Any]) -> tuple[float, ...]:
    out: list[float] = []
    for value in values:
        if isinstance(value, str) and value.strip().lower() in {
            "+inf",
            "inf",
            "infinity",
        }:
            out.append(float("inf"))
        else:
            out.append(float(value))
    if not out or any(np.isnan(item) or item < 0.0 for item in out):
        raise ValueError("alpha_grid must contain non-negative values or +inf")
    return tuple(out)


def _list_override(profile: dict[str, Any], key: str, fallback: Iterable[Any]) -> list[Any]:
    return list(profile[key] if key in profile else fallback)


def _validate_global_seed_namespace(profiles: dict[str, Any]) -> None:
    """Require every declared development and confirmatory base seed to be fresh."""

    used: dict[int, str] = {}
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"profile {profile_name!r} must be a mapping")
        for root in profile.get("roots", []):
            if not isinstance(root, dict):
                raise ValueError(f"profile {profile_name!r} contains a non-mapping root")
            root_id = str(root.get("root_id", ""))
            teacher = root.get("teacher_coupling_seeds", {})
            if not isinstance(teacher, dict):
                raise ValueError(f"{profile_name}.{root_id}: invalid teacher seed map")
            seed_items = [
                (f"{profile_name}.{root_id}.teacher[{n}]", value)
                for n, value in teacher.items()
            ]
            seed_items.extend(
                (f"{profile_name}.{root_id}.{field}", root.get(field))
                for field in _SEED_FIELDS
            )
            for label, value in seed_items:
                seed = _positive_int(label, value, minimum=0)
                if seed in used:
                    raise ValueError(
                        f"global seed collision: {label} and {used[seed]} both use {seed}"
                    )
                used[seed] = label


def load_canonical_profile(config_path: Path, profile_name: str) -> dict[str, Any]:
    """Load and strictly validate one smoke, calibration, or confirmatory profile."""

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("canonical config must have schema_version: 1")
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError(f"unknown canonical profile: {profile_name!r}")
    _validate_global_seed_namespace(profiles)
    common = raw.get("common")
    profile = profiles[profile_name]
    if not isinstance(common, dict) or not isinstance(profile, dict):
        raise ValueError("common and selected profile must be mappings")

    teacher = dict(common["teacher"])
    student = dict(common["student"])
    input_cfg = dict(common["input"])
    fit_cfg = dict(common["fit"])
    decision = dict(common["decision"])
    required_decision = {
        "tolerance_nrmse",
        "min_target_variance",
        "min_root_successes",
            "roots_total",
            "min_spearman",
            "min_g2_live_systems_per_family",
        "min_window_gain",
        "min_feature_gain",
        "min_best_classical_gain",
        "min_scaling_r_squared",
        "g4_gammas",
        "g5_gamma",
    }
    missing_decision = sorted(required_decision.difference(decision))
    if missing_decision:
        raise ValueError(f"common.decision is missing keys: {missing_decision}")
    roots = list(profile.get("roots", []))

    teacher_ns = [
        _positive_int("teacher_n_qubits", item)
        for item in _list_override(profile, "teacher_n_qubits", teacher["n_qubits"])
    ]
    teacher_dissipators = [
        str(item)
        for item in _list_override(
            profile,
            "teacher_dissipators",
            teacher["dissipators"],
        )
    ]
    teacher_gammas = [
        float(item)
        for item in _list_override(profile, "teacher_gammas", teacher["gammas"])
    ]
    windows = [
        _positive_int("window", item)
        for item in _list_override(profile, "windows", student["windows"])
    ]
    snapshot_budgets = [
        _positive_int("snapshot_budget", item)
        for item in _list_override(
            profile,
            "snapshot_budgets",
            student["snapshot_budgets"],
        )
    ]
    max_snapshots = _positive_int("max_snapshots", student["max_snapshots"])
    student_n = _positive_int("student_n_qubits", student["n_qubits"])

    if len(set(teacher_ns)) != len(teacher_ns):
        raise ValueError("teacher_n_qubits values must be unique")
    if any(name not in _DISSIPATORS for name in teacher_dissipators):
        raise ValueError("unknown teacher dissipator")
    if len(set(teacher_dissipators)) != len(teacher_dissipators):
        raise ValueError("teacher dissipators must be unique")
    if any(gamma < 0.0 or not np.isfinite(gamma) for gamma in teacher_gammas):
        raise ValueError("teacher gammas must be finite and non-negative")
    if len(set(windows)) != len(windows) or max(windows) > student_n:
        raise ValueError("windows must be unique and no larger than student_n_qubits")
    if any(
        budget > max_snapshots or max_snapshots % budget != 0
        for budget in snapshot_budgets
    ):
        raise ValueError("snapshot budgets must divide max_snapshots")
    if student.get("architecture") != "ptd_parallel" or student.get("encoding") != "parallel":
        raise ValueError("primary student must be the canonical ptd_parallel encoder")
    if float(student.get("gamma", 0.0)) != 0.0:
        raise ValueError("primary canonical student is frozen as unitary")

    confirmatory = bool(profile.get("confirmatory", False))
    status = str(raw.get("status", ""))
    freeze_record_name = raw.get("freeze_record")
    if not isinstance(freeze_record_name, str) or not freeze_record_name.strip():
        raise ValueError("canonical config must name a freeze_record")
    freeze_record_path = (path.resolve().parent / freeze_record_name).resolve()
    if confirmatory and status != "FROZEN_CONFIRMATORY":
        raise RuntimeError(
            "confirmatory roots are sealed until config status is FROZEN_CONFIRMATORY"
        )
    if not roots:
        raise ValueError("profile must contain at least one root")
    if confirmatory and len(roots) != int(decision["roots_total"]):
        raise ValueError("confirmatory root count does not match roots_total")

    root_ids: set[str] = set()
    used_seeds: dict[int, str] = {}
    for root in roots:
        if not isinstance(root, dict):
            raise ValueError("each root must be a mapping")
        root_id = str(root.get("root_id", ""))
        if not root_id or root_id in root_ids:
            raise ValueError("root_id values must be non-empty and unique")
        root_ids.add(root_id)
        teacher_seeds = root.get("teacher_coupling_seeds")
        if not isinstance(teacher_seeds, dict):
            raise ValueError(f"{root_id}: teacher_coupling_seeds must be a mapping")
        if set(map(str, teacher_seeds)) != set(map(str, teacher_ns)):
            raise ValueError(f"{root_id}: teacher seed keys must match teacher sizes")
        seed_items = [
            (f"{root_id}.teacher[{n}]", teacher_seeds[str(n)])
            for n in teacher_ns
        ]
        seed_items.extend((f"{root_id}.{field}", root.get(field)) for field in _SEED_FIELDS)
        for label, value in seed_items:
            seed = _positive_int(label, value, minimum=0)
            if seed in used_seeds:
                raise ValueError(
                    f"seed collision: {label} and {used_seeds[seed]} both use {seed}"
                )
            used_seeds[seed] = label

    rows = {
        split: _positive_int(f"{split}_rows", profile[f"{split}_rows"])
        for split in ("train", "val", "test")
    }
    prefix_burnin = _positive_int(
        "prefix_burnin",
        profile.get("prefix_burnin", common["suffix_ensemble"]["prefix_burnin"]),
    )
    levels = _positive_int("quantize_levels", input_cfg["quantize_levels"], minimum=2)
    if input_cfg.get("distribution") != "uniform_discrete":
        raise ValueError("canonical study requires uniform_discrete inputs")
    suffix_groups = _positive_int("suffix_groups", profile["suffix_groups"])
    if suffix_groups > levels:
        raise ValueError(
            "suffix_groups cannot exceed quantize_levels because suffix rows "
            "must be unique at every registered window"
        )

    roots_total = _positive_int("decision.roots_total", decision["roots_total"])
    min_root_successes = _positive_int(
        "decision.min_root_successes",
        decision["min_root_successes"],
    )
    if min_root_successes > roots_total:
        raise ValueError("decision.min_root_successes cannot exceed roots_total")
    if not isinstance(decision["g4_gammas"], (list, tuple)):
        raise ValueError("decision.g4_gammas must be a list")
    checked_decision = {
        "tolerance_nrmse": float(decision["tolerance_nrmse"]),
        "min_target_variance": float(decision["min_target_variance"]),
        "min_root_successes": min_root_successes,
        "roots_total": roots_total,
        "min_spearman": float(decision["min_spearman"]),
        "min_g2_live_systems_per_family": _positive_int(
            "decision.min_g2_live_systems_per_family",
            decision["min_g2_live_systems_per_family"],
        ),
        "min_window_gain": float(decision["min_window_gain"]),
        "min_feature_gain": float(decision["min_feature_gain"]),
        "min_best_classical_gain": float(decision["min_best_classical_gain"]),
        "min_scaling_r_squared": float(decision["min_scaling_r_squared"]),
        "g4_gammas": [float(value) for value in decision["g4_gammas"]],
        "g5_gamma": float(decision["g5_gamma"]),
    }
    if (
        not np.isfinite(checked_decision["tolerance_nrmse"])
        or checked_decision["tolerance_nrmse"] <= 0.0
        or not np.isfinite(checked_decision["min_target_variance"])
        or checked_decision["min_target_variance"] <= 0.0
    ):
        raise ValueError("decision error thresholds must be finite and positive")
    gain_keys = (
        "min_window_gain",
        "min_feature_gain",
        "min_best_classical_gain",
    )
    if any(
        not np.isfinite(checked_decision[key])
        or not 0.0 <= checked_decision[key] <= 1.0
        for key in gain_keys
    ):
        raise ValueError("decision gain thresholds must be in [0, 1]")
    if not 0.0 <= checked_decision["min_spearman"] <= 1.0:
        raise ValueError("decision.min_spearman must be in [0, 1]")
    if checked_decision["min_g2_live_systems_per_family"] > len(teacher["gammas"]):
        raise ValueError(
            "decision.min_g2_live_systems_per_family cannot exceed gamma count"
        )
    if not 0.0 <= checked_decision["min_scaling_r_squared"] <= 1.0:
        raise ValueError("decision.min_scaling_r_squared must be in [0, 1]")
    g4_gammas = checked_decision["g4_gammas"]
    common_gammas = {float(value) for value in teacher["gammas"]}
    if (
        not g4_gammas
        or len(set(g4_gammas)) != len(g4_gammas)
        or any(not np.isfinite(value) or value <= 0.0 for value in g4_gammas)
        or not set(g4_gammas) <= common_gammas
    ):
        raise ValueError("decision.g4_gammas must be unique positive teacher gammas")
    if (
        not np.isfinite(checked_decision["g5_gamma"])
        or checked_decision["g5_gamma"] <= 0.0
        or checked_decision["g5_gamma"] not in common_gammas
    ):
        raise ValueError("decision.g5_gamma must be a positive teacher gamma")

    return {
        "schema_version": 1,
        "study": str(raw["study"]),
        "status": status,
        "profile": str(profile_name),
        "confirmatory": confirmatory,
        "config_path": str(path.resolve()),
        "freeze_record_path": str(freeze_record_path),
        "teacher_n_qubits": teacher_ns,
        "teacher_dissipators": teacher_dissipators,
        "teacher_gammas": teacher_gammas,
        "teacher": teacher,
        "student": student,
        "windows": windows,
        "snapshot_budgets": snapshot_budgets,
        "max_snapshots": max_snapshots,
        "quantize_levels": levels,
        "alpha_grid": _float_grid(fit_cfg["alpha_grid"]),
        "rows": rows,
        "suffix_groups": suffix_groups,
        "prefix_branches": _positive_int("prefix_branches", profile["prefix_branches"], minimum=2),
        "prefix_burnin": prefix_burnin,
        "decision": checked_decision,
        "tolerance_nrmse": checked_decision["tolerance_nrmse"],
        "min_target_variance": checked_decision["min_target_variance"],
        "roots": roots,
    }


def _derived_rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, keys)]))


def _inputs(seed: int, length: int, levels: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(levels), size=int(length)).astype(float) / (levels - 1)


def _input_blocks(root: dict[str, Any], profile: dict[str, Any]) -> dict[str, np.ndarray]:
    burnin = int(profile["prefix_burnin"])
    return {
        split: _inputs(
            int(root[f"{split}_input_seed"]),
            burnin + int(profile["rows"][split]),
            int(profile["quantize_levels"]),
        )
        for split in ("train", "val", "test")
    }


def _delay_blocks(
    inputs: dict[str, np.ndarray],
    *,
    window: int,
    burnin: int,
) -> dict[str, np.ndarray]:
    return {
        split: qelm.delay_embed(values, int(window))[int(burnin) :]
        for split, values in inputs.items()
    }


def _teacher_observables(profile: dict[str, Any], n_qubits: int) -> list[readout.Observable]:
    spec = profile["teacher"]["target_observables"]
    return readout.pauli_observables(
        int(n_qubits),
        axes=tuple(spec["axes"]),
        max_weight=int(spec["max_weight"]),
    )


def _student_observables(profile: dict[str, Any]) -> list[readout.Observable]:
    spec = profile["student"]["observables"]
    return readout.pauli_observables(
        int(profile["student"]["n_qubits"]),
        axes=tuple(spec["axes"]),
        max_weight=int(spec["max_weight"]),
    )


def _teacher_reservoir(
    profile: dict[str, Any],
    *,
    n_qubits: int,
    dissipator: str,
    gamma: float,
    coupling_seed: int,
) -> reservoirs.LindbladReservoir:
    cfg = profile["teacher"]
    rng = np.random.default_rng(int(coupling_seed))
    J = reservoirs.random_couplings(int(n_qubits), float(cfg["J_s"]), rng)
    H_static = reservoirs.ising_xx_hamiltonian(J, float(cfg["h"]), int(n_qubits))
    H_drive = float(cfg["h_in"]) * reservoirs.transverse_drive(int(n_qubits))
    jumps = (
        []
        if float(gamma) == 0.0
        else _DISSIPATORS[str(dissipator)](int(n_qubits), float(gamma))
    )
    return reservoirs.LindbladReservoir.from_terms(
        int(n_qubits),
        H_static,
        H_drive,
        jumps,
        float(cfg["dt"]),
        cache_propagators=True,
        quantize=int(profile["quantize_levels"]),
    )


def _student_channel(
    profile: dict[str, Any],
    coupling_seed: int,
) -> qelm.UnitaryQELMChannel:
    cfg = profile["student"]
    n_qubits = int(cfg["n_qubits"])
    rng = np.random.default_rng(int(coupling_seed))
    J = reservoirs.random_couplings(n_qubits, float(cfg["J_s"]), rng)
    H = reservoirs.ising_xx_hamiltonian(J, float(cfg["h"]), n_qubits)
    return qelm.UnitaryQELMChannel.from_hamiltonian(
        n_qubits=n_qubits,
        H_static=H,
        T_evolve=float(cfg["T_evolve"]),
        n_snapshots=int(profile["max_snapshots"]),
    )


def _feature_from_state(
    state: np.ndarray,
    observable_matrices: np.ndarray,
) -> np.ndarray:
    return np.real(np.einsum("kij,ji->k", observable_matrices, state))


def _teacher_blocks(
    reservoir: reservoirs.LindbladReservoir,
    inputs: dict[str, np.ndarray],
    observables: list[readout.Observable],
    *,
    burnin: int,
) -> dict[str, np.ndarray]:
    return {
        split: reservoir.run(
            values,
            observables,
            washout=int(burnin),
            n_virtual=1,
        )
        for split, values in inputs.items()
    }


def _reset_replay_blocks(
    reservoir: reservoirs.LindbladReservoir,
    delay_blocks: dict[str, np.ndarray],
    observables: list[readout.Observable],
) -> dict[str, np.ndarray]:
    matrices = np.stack([item.matrix for item in observables])
    out: dict[str, np.ndarray] = {}
    for split, windows in delay_blocks.items():
        features = np.empty((len(windows), len(observables)), dtype=float)
        for row_idx, newest_first in enumerate(windows):
            state = qelm.time_delayed_window_states(
                reservoir,
                newest_first[::-1],
                1,
            )[-1]
            features[row_idx] = _feature_from_state(state, matrices)
        out[split] = features
    return out


def _stack_blocks(
    blocks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(blocks["train"], dtype=float)
    val = np.asarray(blocks["val"], dtype=float)
    test = np.asarray(blocks["test"], dtype=float)
    values = np.concatenate([train, val, test], axis=0)
    train_idx = np.arange(len(train), dtype=int)
    val_idx = np.arange(len(train), len(train) + len(val), dtype=int)
    test_idx = np.arange(len(train) + len(val), len(values), dtype=int)
    return values, train_idx, val_idx, test_idx


def _standardized_centered_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (values - mean) / scale
    return standardized - np.mean(standardized, axis=0, keepdims=True)


def _resolved_singular_values(features: np.ndarray) -> tuple[np.ndarray, int]:
    centered = _standardized_centered_features(features)
    singular = np.linalg.svd(centered, compute_uv=False)
    if not singular.size or float(singular[0]) == 0.0:
        return singular, 0
    rank = int(np.sum(singular > float(singular[0]) * RIDGE_LSTSQ_RCOND))
    return singular, rank


def _condition_number(features: np.ndarray) -> float:
    singular, rank = _resolved_singular_values(features)
    live = singular[:rank]
    if len(live) == 0:
        return float("inf")
    if len(live) == 1:
        return 1.0
    return float(live[0] / live[-1])


def _fit_model(
    feature_blocks: dict[str, np.ndarray],
    target_blocks: dict[str, np.ndarray],
    alphas: tuple[float, ...],
) -> tuple[MultiOutputRidge, dict[str, Any]]:
    features, train_idx, val_idx, test_idx = _stack_blocks(feature_blocks)
    targets, y_train, y_val, y_test = _stack_blocks(target_blocks)
    if not (
        np.array_equal(train_idx, y_train)
        and np.array_equal(val_idx, y_val)
        and np.array_equal(test_idx, y_test)
    ):
        raise RuntimeError("feature and target split indices differ")
    model = fit_multivariate_ridge(
        features,
        targets,
        train_idx,
        val_idx,
        alphas=alphas,
    )
    metrics = emulation_metrics(targets[test_idx], model.predict(features[test_idx]))
    projection = target_projection_diagnostic(features[train_idx], targets[train_idx])
    return model, {
        **asdict(metrics),
        "alpha": float(model.alpha),
        "intercept_only_selected": bool(np.isposinf(model.alpha)),
        "validation_nrmse": float(model.validation_nrmse),
        "feature_dim": int(features.shape[1]),
        "feature_rank": _resolved_singular_values(features[train_idx])[1],
        "effective_rank": float(
            participation_effective_rank(
                _standardized_centered_features(features[train_idx])
            )
        ),
        "condition_number": _condition_number(features[train_idx]),
        "target_aligned_fraction": float(projection.target_aligned_fraction),
    }


def _direct_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    feature_dim: int,
) -> dict[str, Any]:
    metrics = emulation_metrics(targets, predictions)
    return {
        **asdict(metrics),
        "alpha": None,
        "intercept_only_selected": None,
        "validation_nrmse": None,
        "feature_dim": int(feature_dim),
        "feature_rank": None,
        "effective_rank": None,
        "condition_number": None,
        "target_aligned_fraction": None,
    }


def _root_student_features(
    profile: dict[str, Any],
    root: dict[str, Any],
    inputs: dict[str, np.ndarray],
) -> dict[int, dict[str, Any]]:
    channel = _student_channel(profile, int(root["student_coupling_seed"]))
    observables = _student_observables(profile)
    n_obs = len(observables)
    burnin = int(profile["prefix_burnin"])
    max_width = n_obs * int(profile["max_snapshots"])
    out: dict[int, dict[str, Any]] = {}
    for window in profile["windows"]:
        delays = _delay_blocks(inputs, window=int(window), burnin=burnin)
        qelm_max = {
            split: channel.features(
                values,
                observables,
                encoding_scheme="one_to_one",
            )
            for split, values in delays.items()
        }
        classical = fit_random_tanh_feature_map(
            delays["train"],
            width=max_width,
            seed=int(
                _derived_rng(int(root["classical_feature_seed"]), int(window)).integers(
                    0, 2**32 - 1
                )
            ),
        )
        out[int(window)] = {
            "delays": delays,
            "qelm_max": qelm_max,
            "classical_map": classical,
            "classical_max": {
                split: classical.transform(values)
                for split, values in delays.items()
            },
            "linear": delays,
            "poly2": {
                split: degree_two_delay_features(values)
                for split, values in delays.items()
            },
            "n_student_observables": n_obs,
        }
    return out


def _suffix_inputs(
    profile: dict[str, Any],
    root: dict[str, Any],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    groups = int(profile["suffix_groups"])
    branches = int(profile["prefix_branches"])
    burnin = int(profile["prefix_burnin"])
    levels = int(profile["quantize_levels"])
    prefix_rng = np.random.default_rng(int(root["prefix_input_seed"]))
    prefixes = prefix_rng.integers(
        0,
        levels,
        size=(groups, branches, burnin),
    ).astype(float) / (levels - 1)
    max_window = max(int(window) for window in profile["windows"])
    suffix_rng = np.random.default_rng(int(root["suffix_input_seed"]))
    # Draw the newest symbol without replacement.  Because every registered
    # window is a tail of this panel, that one unique coordinate guarantees
    # unique numerical suffix rows for every d.  Without this condition, two
    # identical suffix values could be assigned separate group means and leak
    # old-prefix variation from the history term into the representation term.
    newest = suffix_rng.permutation(levels)[:groups]
    if max_window > 1:
        older = suffix_rng.integers(
            0,
            levels,
            size=(groups, max_window - 1),
        )
        longest_integer = np.column_stack([older, newest])
    else:
        longest_integer = newest[:, None]
    longest_suffixes = longest_integer.astype(float) / (levels - 1)
    suffixes = {
        int(window): longest_suffixes[:, -int(window) :].copy()
        for window in profile["windows"]
    }
    return prefixes, suffixes


def _prefix_states(
    reservoir: reservoirs.LindbladReservoir,
    prefixes: np.ndarray,
) -> np.ndarray:
    groups, branches, _ = prefixes.shape
    dim = 2 ** reservoir.n_qubits
    states = np.empty((groups, branches, dim, dim), dtype=complex)
    for group in range(groups):
        for branch in range(branches):
            state = reservoir.initial_state()
            for value in prefixes[group, branch]:
                state = reservoir.step(state, float(value))
            states[group, branch] = state
    return states


def _suffix_targets(
    reservoir: reservoirs.LindbladReservoir,
    prefix_states: np.ndarray,
    suffixes: np.ndarray,
    observables: list[readout.Observable],
) -> np.ndarray:
    matrices = np.stack([item.matrix for item in observables])
    groups, branches = prefix_states.shape[:2]
    targets = np.empty((groups, branches, len(observables)), dtype=float)
    for group in range(groups):
        for branch in range(branches):
            state = prefix_states[group, branch].copy()
            for value in suffixes[group]:
                state = reservoir.step(state, float(value))
            targets[group, branch] = _feature_from_state(state, matrices)
    return targets


def _suffix_reset_replay(
    reservoir: reservoirs.LindbladReservoir,
    suffixes: np.ndarray,
    observables: list[readout.Observable],
) -> np.ndarray:
    matrices = np.stack([item.matrix for item in observables])
    predictions = np.empty((len(suffixes), len(observables)), dtype=float)
    for group, suffix in enumerate(suffixes):
        state = reservoir.initial_state()
        for value in suffix:
            state = reservoir.step(state, float(value))
        predictions[group] = _feature_from_state(state, matrices)
    return predictions


def _pair_diagnostics(targets: np.ndarray) -> dict[str, float]:
    groups, branches, _ = targets.shape
    squared: list[float] = []
    group_max: list[float] = []
    for group in range(groups):
        distances: list[float] = []
        for first in range(branches):
            for second in range(first + 1, branches):
                distance = float(
                    np.sqrt(np.mean((targets[group, first] - targets[group, second]) ** 2))
                )
                distances.append(distance)
                squared.append(distance**2)
        group_max.append(max(distances))
    return {
        "mean_group_diameter": float(np.mean(group_max)),
        "max_group_diameter": float(np.max(group_max)),
        "half_max_group_diameter": float(0.5 * np.max(group_max)),
        "rms_pair_half_bound": float(0.5 * np.sqrt(np.mean(squared))),
    }


def _suffix_prediction_audit(
    targets: np.ndarray,
    prediction: np.ndarray,
    repeated_predictions: np.ndarray,
    pair_diagnostics: dict[str, float],
    *,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Audit one deterministic suffix predictor on repeated identical rows.

    ``prediction`` is the ordinary one-prediction-per-suffix production result.
    ``repeated_predictions`` is obtained by separately evaluating the same
    suffix once for every prefix branch.  The two exact common-suffix lower
    bounds are valid only when all of those predictions agree.
    """

    truth = np.asarray(targets, dtype=float)
    base = np.asarray(prediction, dtype=float)
    repeated = np.asarray(repeated_predictions, dtype=float)
    if truth.ndim != 3:
        raise ValueError("targets must have shape (groups, branches, outputs)")
    if base.shape != (truth.shape[0], truth.shape[2]):
        raise ValueError("prediction must have shape (groups, outputs)")
    if repeated.shape != truth.shape:
        raise ValueError("repeated_predictions must have the same shape as targets")
    if not (
        np.all(np.isfinite(truth))
        and np.all(np.isfinite(base))
        and np.all(np.isfinite(repeated))
    ):
        raise ValueError("suffix audit arrays must contain only finite values")
    if not np.isfinite(float(atol)) or float(atol) < 0.0:
        raise ValueError("atol must be finite and non-negative")

    # Include both the ordinary vectorized production prediction and every
    # independently evaluated common-suffix arm in the agreement audit.
    candidates = np.concatenate([base[:, None, :], repeated], axis=1)
    disagreement = 0.0
    for first in range(candidates.shape[1]):
        for second in range(first + 1, candidates.shape[1]):
            pair_rms = np.sqrt(
                np.mean((candidates[:, first] - candidates[:, second]) ** 2, axis=1)
            )
            disagreement = max(disagreement, float(np.max(pair_rms)))

    branch_rmse = np.sqrt(np.mean((truth - repeated) ** 2, axis=2))
    worst_branch_rmse = float(np.max(branch_rmse))
    repeated_rmse = float(np.sqrt(np.mean((truth - repeated) ** 2)))
    rms_bound = float(pair_diagnostics["rms_pair_half_bound"])
    diameter_bound = 0.5 * float(pair_diagnostics["max_group_diameter"])
    rms_slack = repeated_rmse - rms_bound
    diameter_slack = worst_branch_rmse - diameter_bound
    predictions_agree = bool(disagreement <= float(atol))
    rms_satisfied = bool(predictions_agree and rms_slack >= -float(atol))
    diameter_satisfied = bool(
        predictions_agree and diameter_slack >= -float(atol)
    )
    return {
        "prediction_disagreement": disagreement,
        "repeated_suffix_rmse": repeated_rmse,
        "worst_branch_rmse": worst_branch_rmse,
        "rms_bound_slack": float(rms_slack),
        "diameter_bound_slack": float(diameter_slack),
        "predictions_agree": predictions_agree,
        "rms_half_bound_satisfied": rms_satisfied,
        "diameter_half_bound_satisfied": diameter_satisfied,
        "suffix_bounds_satisfied": bool(rms_satisfied and diameter_satisfied),
    }


def _predict_rows_independently(
    model: MultiOutputRidge,
    features: np.ndarray,
) -> np.ndarray:
    """Exercise the production readout once per row for suffix-arm auditing."""

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("features must be a non-empty two-dimensional array")
    return np.stack(
        [model.predict(values[row : row + 1])[0] for row in range(len(values))]
    )


def _system_identifier(root_id: str, n_qubits: int, dissipator: str, gamma: float) -> str:
    gamma_label = f"{float(gamma):.3f}".replace(".", "p")
    return f"{root_id}_n{int(n_qubits)}_{dissipator}_g{gamma_label}"


def _row_base(
    *,
    root_id: str,
    system_id: str,
    n_qubits: int,
    dissipator: str,
    gamma: float,
    teacher_seed: int,
    window: int,
    target_observables_sha256: str,
    student_n_qubits: int,
    student_observables_sha256: str,
) -> dict[str, Any]:
    return {
        "root_id": root_id,
        "system_id": system_id,
        "teacher_n_qubits": int(n_qubits),
        "teacher_dissipator": str(dissipator),
        "teacher_gamma": float(gamma),
        "teacher_coupling_seed": int(teacher_seed),
        "window": int(window),
        "target_observables_sha256": str(target_observables_sha256),
        "student_n_qubits": int(student_n_qubits),
        "student_observables_sha256": str(student_observables_sha256),
    }


def _model_feature_blocks(
    root_features: dict[str, Any],
    model_name: str,
    snapshot_budget: int | None,
    profile: dict[str, Any],
) -> dict[str, np.ndarray]:
    if model_name == "ptd_parallel":
        return {
            split: select_nested_snapshots(
                values,
                max_snapshots=int(profile["max_snapshots"]),
                selected_snapshots=int(snapshot_budget),
                n_observables=int(root_features["n_student_observables"]),
            )
            for split, values in root_features["qelm_max"].items()
        }
    if model_name == "classical_tanh":
        width = int(snapshot_budget) * int(root_features["n_student_observables"])
        return {
            split: values[:, :width]
            for split, values in root_features["classical_max"].items()
        }
    if model_name == "linear_delay":
        return root_features["linear"]
    if model_name == "poly2_delay":
        return root_features["poly2"]
    raise ValueError(f"unknown learned model: {model_name}")


def run_canonical_study(
    *,
    config_path: Path,
    profile_name: str,
    output_dir: Path,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Execute one allowed profile and write strict, auditable artifacts."""

    profile = load_canonical_profile(Path(config_path), profile_name)
    run_started = perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    reproducibility = canonical_reproducibility_snapshot(
        repo_root=repo_root,
        config_path=Path(config_path),
        runner_path=runner_path,
    )
    freeze_audit: dict[str, Any] | None = None
    if profile["confirmatory"]:
        material = reproducibility["material"]
        freeze_audit = verify_canonical_freeze_record(
            Path(profile["freeze_record_path"]),
            study=str(profile["study"]),
            profile=str(profile["profile"]),
            config_sha256=str(material["files"]["configuration"]["sha256"]),
            material_identity_sha256=str(material["identity_sha256"]),
        )
        if not freeze_audit["passed"]:
            raise RuntimeError(
                "confirmatory material does not match the approved freeze record"
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stream_rows: list[dict[str, Any]] = []
    suffix_rows: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []

    for root in profile["roots"]:
        root_id = str(root["root_id"])
        inputs = _input_blocks(root, profile)
        student_features = _root_student_features(profile, root, inputs)
        prefixes, suffixes_by_window = _suffix_inputs(profile, root)
        student_channel = _student_channel(profile, int(root["student_coupling_seed"]))
        student_observables = _student_observables(profile)
        student_observable_hash = observable_names_sha256(
            item.name for item in student_observables
        )

        suffix_root_features: dict[int, dict[str, Any]] = {}
        for window in profile["windows"]:
            suffix_newest_first = suffixes_by_window[int(window)][:, ::-1]
            repeated_suffixes = np.repeat(
                suffix_newest_first[:, None, :],
                int(profile["prefix_branches"]),
                axis=1,
            )
            repeated_flat = repeated_suffixes.reshape(
                -1, repeated_suffixes.shape[-1]
            )
            qelm_max = student_channel.features(
                suffix_newest_first,
                student_observables,
                encoding_scheme="one_to_one",
            )
            classical_map = student_features[int(window)]["classical_map"]
            suffix_root_features[int(window)] = {
                "delays": suffix_newest_first,
                "qelm_max": qelm_max,
                "classical_max": classical_map.transform(suffix_newest_first),
                "poly2": degree_two_delay_features(suffix_newest_first),
                "repeated_shape": repeated_suffixes.shape[:2],
                "repeated_delays": repeated_flat,
                "repeated_qelm_max": student_channel.features(
                    repeated_flat,
                    student_observables,
                    encoding_scheme="one_to_one",
                ),
                "repeated_classical_max": classical_map.transform(repeated_flat),
                "repeated_poly2": degree_two_delay_features(repeated_flat),
                "n_student_observables": len(student_observables),
            }

        for n_qubits in profile["teacher_n_qubits"]:
            teacher_seed = int(root["teacher_coupling_seeds"][str(n_qubits)])
            observables = _teacher_observables(profile, int(n_qubits))
            target_observable_hash = observable_names_sha256(
                item.name for item in observables
            )
            for dissipator in profile["teacher_dissipators"]:
                for gamma in profile["teacher_gammas"]:
                    reservoir = _teacher_reservoir(
                        profile,
                        n_qubits=int(n_qubits),
                        dissipator=str(dissipator),
                        gamma=float(gamma),
                        coupling_seed=teacher_seed,
                    )
                    target_blocks = _teacher_blocks(
                        reservoir,
                        inputs,
                        observables,
                        burnin=int(profile["prefix_burnin"]),
                    )
                    target_variance = float(
                        np.mean(
                            (
                                target_blocks["test"]
                                - np.mean(target_blocks["test"], axis=0, keepdims=True)
                            )
                            ** 2
                        )
                    )
                    system_id = _system_identifier(
                        root_id,
                        int(n_qubits),
                        str(dissipator),
                        float(gamma),
                    )
                    prefix_states = _prefix_states(reservoir, prefixes)
                    system_rows.append(
                        {
                            "root_id": root_id,
                            "system_id": system_id,
                            "teacher_n_qubits": int(n_qubits),
                            "teacher_dissipator": str(dissipator),
                            "teacher_gamma": float(gamma),
                            "teacher_coupling_seed": teacher_seed,
                            "target_dim": len(observables),
                            "target_observables_sha256": target_observable_hash,
                            "target_variance": target_variance,
                            "target_live": bool(
                                target_variance >= float(profile["min_target_variance"])
                            ),
                        }
                    )

                    for window in profile["windows"]:
                        window = int(window)
                        root_window = student_features[window]
                        delay_blocks = root_window["delays"]
                        reset_blocks = _reset_replay_blocks(
                            reservoir,
                            delay_blocks,
                            observables,
                        )
                        base = _row_base(
                            root_id=root_id,
                            system_id=system_id,
                            n_qubits=int(n_qubits),
                            dissipator=str(dissipator),
                            gamma=float(gamma),
                            teacher_seed=teacher_seed,
                            window=window,
                            target_observables_sha256=target_observable_hash,
                            student_n_qubits=int(profile["student"]["n_qubits"]),
                            student_observables_sha256=student_observable_hash,
                        )
                        learned_models: dict[tuple[str, int | None], MultiOutputRidge] = {}

                        stream_rows.append(
                            {
                                **base,
                                "model": "rr_same_identity",
                                "snapshot_budget": None,
                                "feature_budget": len(observables),
                                **_direct_metrics(
                                    reset_blocks["test"],
                                    target_blocks["test"],
                                    feature_dim=len(observables),
                                ),
                            }
                        )
                        constant = np.mean(target_blocks["train"], axis=0)
                        constant_test = np.repeat(
                            constant[None, :],
                            len(target_blocks["test"]),
                            axis=0,
                        )
                        stream_rows.append(
                            {
                                **base,
                                "model": "constant",
                                "snapshot_budget": None,
                                "feature_budget": 0,
                                **_direct_metrics(
                                    constant_test,
                                    target_blocks["test"],
                                    feature_dim=0,
                                ),
                            }
                        )

                        learned_specs: list[tuple[str, int | None]] = [
                            ("linear_delay", None),
                            ("poly2_delay", None),
                        ]
                        learned_specs.extend(
                            (model_name, int(budget))
                            for model_name in ("ptd_parallel", "classical_tanh")
                            for budget in profile["snapshot_budgets"]
                        )
                        for model_name, budget in learned_specs:
                            blocks = _model_feature_blocks(
                                root_window,
                                model_name,
                                budget,
                                profile,
                            )
                            model, metrics = _fit_model(
                                blocks,
                                target_blocks,
                                profile["alpha_grid"],
                            )
                            learned_models[(model_name, budget)] = model
                            stream_rows.append(
                                {
                                    **base,
                                    "model": model_name,
                                    "snapshot_budget": budget,
                                    "feature_budget": int(blocks["train"].shape[1]),
                                    **metrics,
                                }
                            )

                        suffixes = suffixes_by_window[window]
                        suffix_targets = _suffix_targets(
                            reservoir,
                            prefix_states,
                            suffixes,
                            observables,
                        )
                        rr_suffix = _suffix_reset_replay(
                            reservoir,
                            suffixes,
                            observables,
                        )
                        repeated_suffixes = np.repeat(
                            suffixes[:, None, :],
                            int(profile["prefix_branches"]),
                            axis=1,
                        )
                        rr_suffix_repeated = _suffix_reset_replay(
                            reservoir,
                            repeated_suffixes.reshape(
                                -1, repeated_suffixes.shape[-1]
                            ),
                            observables,
                        ).reshape(suffix_targets.shape)
                        oracle_decomposition = suffix_risk_decomposition(
                            suffix_targets,
                            rr_suffix,
                        )
                        pair_diag = _pair_diagnostics(suffix_targets)

                        suffix_predictions: list[
                            tuple[
                                str,
                                int | None,
                                int,
                                np.ndarray,
                                np.ndarray,
                            ]
                        ] = [
                            (
                                "rr_same_identity",
                                None,
                                len(observables),
                                rr_suffix,
                                rr_suffix_repeated,
                            ),
                            (
                                "constant",
                                None,
                                0,
                                np.repeat(constant[None, :], len(suffixes), axis=0),
                                np.broadcast_to(
                                    constant,
                                    suffix_targets.shape,
                                ).copy(),
                            ),
                        ]
                        suffix_features = suffix_root_features[window]
                        for (model_name, budget), model in learned_models.items():
                            if model_name == "ptd_parallel":
                                features = select_nested_snapshots(
                                    suffix_features["qelm_max"],
                                    max_snapshots=int(profile["max_snapshots"]),
                                    selected_snapshots=int(budget),
                                    n_observables=int(
                                        suffix_features["n_student_observables"]
                                    ),
                                )
                                repeated_features = select_nested_snapshots(
                                    suffix_features["repeated_qelm_max"],
                                    max_snapshots=int(profile["max_snapshots"]),
                                    selected_snapshots=int(budget),
                                    n_observables=int(
                                        suffix_features["n_student_observables"]
                                    ),
                                )
                            elif model_name == "classical_tanh":
                                width = int(budget) * int(
                                    suffix_features["n_student_observables"]
                                )
                                features = suffix_features["classical_max"][:, :width]
                                repeated_features = suffix_features[
                                    "repeated_classical_max"
                                ][:, :width]
                            elif model_name == "linear_delay":
                                features = suffix_features["delays"]
                                repeated_features = suffix_features[
                                    "repeated_delays"
                                ]
                            elif model_name == "poly2_delay":
                                features = suffix_features["poly2"]
                                repeated_features = suffix_features[
                                    "repeated_poly2"
                                ]
                            else:
                                raise RuntimeError("unexpected learned model")
                            repeated_prediction = _predict_rows_independently(
                                model,
                                repeated_features,
                            ).reshape(
                                *suffix_features["repeated_shape"],
                                -1,
                            )
                            production_prediction = model.predict(features)
                            suffix_predictions.append(
                                (
                                    model_name,
                                    budget,
                                    int(features.shape[1]),
                                    production_prediction,
                                    repeated_prediction,
                                )
                            )

                        for (
                            model_name,
                            budget,
                            feature_budget,
                            prediction,
                            repeated_prediction,
                        ) in suffix_predictions:
                            decomposition = suffix_risk_decomposition(
                                suffix_targets,
                                prediction,
                            )
                            prediction_audit = _suffix_prediction_audit(
                                suffix_targets,
                                prediction,
                                repeated_prediction,
                                pair_diag,
                            )
                            suffix_rows.append(
                                {
                                    **base,
                                    "model": model_name,
                                    "snapshot_budget": budget,
                                    "feature_budget": int(feature_budget),
                                    **asdict(decomposition),
                                    "history_nrmse": decomposition.history_nrmse,
                                    "representation_nrmse": decomposition.representation_nrmse,
                                    "total_nrmse": decomposition.total_nrmse,
                                    "history_fraction": decomposition.history_fraction,
                                    "oracle_nrmse": oracle_decomposition.total_nrmse,
                                    "regime": classify_emulation_regime(
                                        history_nrmse=decomposition.history_nrmse,
                                        student_nrmse=decomposition.total_nrmse,
                                        oracle_nrmse=oracle_decomposition.total_nrmse,
                                        target_variance=decomposition.target_variance,
                                        tolerance=float(profile["tolerance_nrmse"]),
                                        min_target_variance=float(
                                            profile["min_target_variance"]
                                        ),
                                    ),
                                    **pair_diag,
                                    **prediction_audit,
                                }
                            )

    stream = pd.DataFrame(stream_rows)
    suffix = pd.DataFrame(suffix_rows)
    systems = pd.DataFrame(system_rows)
    resources = canonical_resource_report(
        profile,
        wall_clock_seconds=perf_counter() - run_started,
    )
    stream.to_csv(output / "stream_metrics.csv", index=False)
    suffix.to_csv(output / "suffix_decomposition.csv", index=False)
    systems.to_csv(output / "system_diagnostics.csv", index=False)

    ptd_suffix = suffix[suffix["model"] == "ptd_parallel"]
    summary = {
        "schema_version": 1,
        "study": profile["study"],
        "profile": profile["profile"],
        "confirmatory": profile["confirmatory"],
        "status": profile["status"],
        "decision": "CALIBRATION_ONLY" if not profile["confirmatory"] else "PENDING_GATES",
        "counts": {
            "roots": int(systems["root_id"].nunique()),
            "systems": int(len(systems)),
            "stream_rows": int(len(stream)),
            "suffix_rows": int(len(suffix)),
            "target_live_systems": int(systems["target_live"].sum()),
        },
        "ptd_regime_counts": {
            str(key): int(value)
            for key, value in ptd_suffix["regime"].value_counts().to_dict().items()
        },
        "max_decomposition_identity_error": float(
            np.max(np.abs(suffix["identity_error"]))
        ),
    }
    _write_json(output / "canonical_summary.json", summary)
    _write_json(output / "computational_resources.json", resources)

    material_files = reproducibility["material"]["files"]
    manifest = {
        "schema_version": 1,
        "study": profile["study"],
        "profile": profile["profile"],
        "confirmatory": profile["confirmatory"],
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": material_files["configuration"]["sha256"],
        "module_sha256": material_files[
            "source:src/qrc_vs_qelm/canonical_study.py"
        ]["sha256"],
        "runner_sha256": material_files.get("runner", {}).get("sha256"),
        "git": reproducibility["git"],
        "reproducibility": reproducibility,
        "freeze_record_audit": freeze_audit,
        "artifacts": {
            name: sha256((output / name).read_bytes()).hexdigest()
            for name in (
                "stream_metrics.csv",
                "suffix_decomposition.csv",
                "system_diagnostics.csv",
                "canonical_summary.json",
                "computational_resources.json",
            )
        },
    }
    _write_json(output / "canonical_manifest.json", manifest)
    return summary


def _git_state(repo_root: Path) -> dict[str, Any]:
    return git_worktree_manifest(repo_root)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
