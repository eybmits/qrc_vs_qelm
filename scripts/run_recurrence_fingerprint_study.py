#!/usr/bin/env python
"""Locked, architecture-blind recurrence-fingerprint validation study.

The collector gives the classifier anonymous case identifiers and a public
maximum explicit-window scope only.  Source labels and actual QELM windows are
kept in a separate key, predictions are serialized and hashed before the key is
merged, and every finite-shot seed is classified as an independent experiment.

This runner is separate from the original phase-map preregistration.  It does
not alter or reuse the original confirmatory seed grids.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrc_vs_qelm import manifests, pipeline, task_families
from qrc_vs_qelm.core import readout
from qrc_vs_qelm.matching import assert_matched
from qrc_vs_qelm.models.api import (
    QRCConfig,
    build_features,
    windowed_qelm_from_qrc,
)
from qrc_vs_qelm.recurrence_fingerprint import (
    FingerprintRule,
    classify_blind,
)


INFINITY = "infinity"
V1_FREEZE_NAME = "recurrence_fingerprint_v1"
V1_CONFIG_SHA256 = "9098b7d6fdf55381d8179321b2a5fcfd5c3821be1df79163424687beb41cac44"
FORBIDDEN_CLASSIFIER_COLUMNS = {
    "curve_model",
    "source_model",
    "source_family",
    "expected_behavior",
    "actual_window",
    "adapter_type",
}


def _load_poc_module():
    path = ROOT / "scripts" / "recurrence_fingerprint_poc.py"
    spec = importlib.util.spec_from_file_location("recurrence_fingerprint_poc_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import exploratory physics helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


POC = _load_poc_module()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _blind_id(lock_hash: str, physics_id: str, source_model: str) -> str:
    payload = f"{lock_hash}|{physics_id}|{source_model}".encode("utf-8")
    return f"case_{hashlib.sha256(payload).hexdigest()[:16]}"


def load_locked_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lock_hash = manifests.sha256_file(path)
    required = {
        "schema_version",
        "freeze_name",
        "scope",
        "probe",
        "measurement",
        "classifier",
        "task_link",
        "design_blocks",
        "heldout_blocks",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"locked config is missing keys: {sorted(missing)}")
    design_ids = {str(row["physics_id"]) for row in payload["design_blocks"]}
    heldout_ids = {str(row["physics_id"]) for row in payload["heldout_blocks"]}
    overlap = design_ids & heldout_ids
    if overlap:
        raise ValueError(f"design/heldout physics overlap: {sorted(overlap)}")
    design_seeds = {int(row["coupling_seed"]) for row in payload["design_blocks"]}
    heldout_seeds = {int(row["coupling_seed"]) for row in payload["heldout_blocks"]}
    if design_seeds & heldout_seeds:
        raise ValueError("design and heldout coupling seeds must be disjoint")
    if int(payload["scope"]["max_window"]) != 5:
        raise ValueError("v1 study is locked to max_window=5")
    if (
        str(payload["freeze_name"]) == V1_FREEZE_NAME
        and lock_hash != V1_CONFIG_SHA256
    ):
        raise ValueError(
            "recurrence_fingerprint_v1 config does not match its frozen SHA-256"
        )
    return payload, lock_hash


def _study_source_files(config_path: Path) -> dict[str, Path]:
    files = {
        "runner": Path(__file__).resolve(),
        "classifier": ROOT / "src" / "qrc_vs_qelm" / "recurrence_fingerprint.py",
        "physics_poc": ROOT / "scripts" / "recurrence_fingerprint_poc.py",
        "locked_config": config_path.resolve(),
        "human_protocol": ROOT / "RECURRENCE_FINGERPRINT_PREREGISTRATION.md",
        "dependency_manifest": ROOT / "pyproject.toml",
        "runner_tests": ROOT / "tests" / "test_recurrence_fingerprint_study.py",
        "classifier_tests": ROOT / "tests" / "test_recurrence_fingerprint.py",
    }
    package_root = ROOT / "src" / "qrc_vs_qelm"
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        files.setdefault(f"package:{relative}", path)
    return files


def _write_opening_manifest(
    *,
    out_dir: Path,
    split: str,
    config_path: Path,
    lock_hash: str,
    allow_dirty: bool,
) -> Path:
    """Seal code/config provenance before any split-specific curve is generated."""
    if split == "heldout" and out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            "heldout output directory is not empty; the open-once split cannot "
            "be regenerated in place"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        name: manifests.sha256_file(path)
        for name, path in _study_source_files(config_path).items()
        if path.exists()
    }
    return manifests.write_manifest(
        out_dir,
        "recurrence_fingerprint_opening",
        {
            "repo_root": ROOT,
            "study": "recurrence_fingerprint_v1",
            "split": split,
            "opened_before_collection": True,
            "allow_dirty": bool(allow_dirty),
            "config_sha256": lock_hash,
            "source_sha256": source_hashes,
        },
    )


def _validate_opening_manifest(
    *,
    opening_manifest_path: Path,
    config_path: Path,
    lock_hash: str,
) -> dict[str, Any]:
    opening = json.loads(opening_manifest_path.read_text(encoding="utf-8"))
    payload = opening["core"]["payload"]
    current_hashes = {
        name: manifests.sha256_file(path)
        for name, path in _study_source_files(config_path).items()
        if path.exists()
    }
    checks = {
        "config_hash_matches_lock": str(payload["config_sha256"]) == lock_hash,
        "config_unchanged_since_opening": manifests.sha256_file(config_path)
        == lock_hash,
        "source_hashes_unchanged_since_opening": payload["source_sha256"]
        == current_hashes,
        "opened_before_collection_recorded": bool(
            payload["opened_before_collection"]
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            f"opening provenance seal failure: {', '.join(failed)}"
        )
    return {
        "pass": True,
        "checks": checks,
        "opening_manifest_sha256": manifests.sha256_file(
            opening_manifest_path
        ),
        "source_sha256": current_hashes,
    }


def _profile_for_block(config: dict[str, Any], block: dict[str, Any]):
    probe = config["probe"]
    max_window = int(config["scope"]["max_window"])
    n_qubits = int(block["n_qubits"])
    specs = (
        POC.QELMSpec("time_delayed", 1),
        POC.QELMSpec("time_delayed", 3),
        POC.QELMSpec("time_delayed", max_window),
        POC.QELMSpec("spatial", min(n_qubits, 3)),
    )
    return replace(
        POC.PROFILES["smoke"],
        name=f"locked_{block['physics_id']}",
        n_qubits=n_qubits,
        gamma=float(block["gamma"]),
        dt=float(block["dt"]),
        h=float(block.get("h", 0.5)),
        h_in=float(block.get("h_in", 1.0)),
        J_s=float(block.get("J_s", 1.0)),
        dissipator=str(block["dissipator"]),
        quantize_levels=int(block.get("quantize_levels", 33)),
        burnin_length=int(probe["burnin_length"]),
        branch_length=int(probe["branch_length"]),
        suffix_lengths=tuple(int(x) for x in probe["suffix_lengths"]),
        pairs_per_coupling=int(probe["histories_per_system"]),
        coupling_seeds=(int(block["coupling_seed"]),),
        history_seed=int(block["history_seed"]),
        shot_seed=0,
        shot_tiers=(None,),
        n_boot=32,
        qelm_specs=specs,
        reference_window=max_window,
    )


def _source_metadata(profile) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = [
        {
            "source_model": "qrc",
            "curve_model": "qrc",
            "source_family": "qrc",
            "expected_behavior": "recurrent_reset_sensitive",
            "actual_window": None,
            "adapter_type": "honest",
            "primary_case": True,
        }
    ]
    for spec in profile.qelm_specs:
        source_model = (
            "spatial_qelm_dmin"
            if getattr(spec, "encoding", None) == "spatial"
            else spec.model_name
        )
        sources.append(
            {
                "source_model": source_model,
                "curve_model": spec.model_name,
                "source_family": "qelm",
                "expected_behavior": "finite_window",
                "actual_window": int(spec.window),
                "adapter_type": "honest",
                "primary_case": True,
            }
        )
    sources.extend(
        [
            {
                "source_model": "forced_reset_qrc_d3",
                "curve_model": "forced_reset_qrc_d3",
                "source_family": "fault_control",
                "expected_behavior": "finite_window",
                "actual_window": 3,
                "adapter_type": "forced_state_erasure",
                "primary_case": False,
            },
            {
                "source_model": "leaky_qelm_control",
                "curve_model": "leaky_qelm_control",
                "source_family": "fault_control",
                "expected_behavior": "recurrent_reset_sensitive",
                "actual_window": None,
                "adapter_type": "state_leak",
                "primary_case": False,
            },
            {
                "source_model": "reset_failure_qrc_control",
                "curve_model": "reset_failure_qrc_control",
                "source_family": "fault_control",
                "expected_behavior": "recurrent_reset_insensitive",
                "actual_window": None,
                "adapter_type": "reset_noop",
                "primary_case": False,
            },
            {
                "source_model": "dead_readout_control",
                "curve_model": "dead_readout_control",
                "source_family": "fault_control",
                "expected_behavior": "low_signal",
                "actual_window": None,
                "adapter_type": "constant_output",
                "primary_case": False,
            },
        ]
    )
    return sources


def _augmented_curve_pairs(
    base: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
    suffix_lengths: Iterable[int],
) -> dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]]:
    records = dict(base)
    for length in suffix_lengths:
        qrc_carry = base[("qrc", "carry", int(length))]
        qrc_reset = base[("qrc", "ideal_quantum_reset", int(length))]

        # Same QRC physics with a deterministic state erasure after the third
        # common input.  This is a behavioral finite-memory control, not a new
        # architecture.
        forced_carry = qrc_carry if int(length) < 3 else qrc_reset
        records[("forced_reset_qrc_d3", "carry", int(length))] = (
            forced_carry[0].copy(),
            forced_carry[1].copy(),
        )
        records[("forced_reset_qrc_d3", "ideal_quantum_reset", int(length))] = (
            qrc_reset[0].copy(),
            qrc_reset[1].copy(),
        )

        # Deliberate row-to-row state leakage.  Copying the recurrent trajectory
        # makes the injected fault strength exact and reproducible.
        records[("leaky_qelm_control", "carry", int(length))] = (
            qrc_carry[0].copy(),
            qrc_carry[1].copy(),
        )
        records[("leaky_qelm_control", "ideal_quantum_reset", int(length))] = (
            qrc_reset[0].copy(),
            qrc_reset[1].copy(),
        )

        # A reset instruction that is silently ignored must leave the tail in
        # both conditions and therefore fail the reset-sensitive criterion.
        records[("reset_failure_qrc_control", "carry", int(length))] = (
            qrc_carry[0].copy(),
            qrc_carry[1].copy(),
        )
        records[(
            "reset_failure_qrc_control",
            "ideal_quantum_reset",
            int(length),
        )] = (qrc_carry[0].copy(), qrc_carry[1].copy())

        zero_a = np.zeros_like(qrc_carry[0])
        zero_b = np.zeros_like(qrc_carry[1])
        for condition in ("carry", "ideal_quantum_reset"):
            records[("dead_readout_control", condition, int(length))] = (
                zero_a.copy(),
                zero_b.copy(),
            )
    return records


def _collect_block(
    config: dict[str, Any],
    lock_hash: str,
    block: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = _profile_for_block(config, block)
    reservoir, channel, observables = POC._build_dynamics(
        profile,
        int(block["coupling_seed"]),
    )
    sources = _source_metadata(profile)
    source_by_curve = {str(row["curve_model"]): row for row in sources}
    source_codes = {name: index for index, name in enumerate(source_by_curve)}
    condition_codes = {"carry": 0, "ideal_quantum_reset": 1}
    suffix_codes = {value: index for index, value in enumerate(profile.suffix_lengths)}
    measurement = config["measurement"]
    shot_tiers = tuple(int(value) for value in measurement["shots_per_batch"])
    measurement_seeds = tuple(int(value) for value in measurement["measurement_seeds"])

    key_rows: list[dict[str, Any]] = []
    for source in sources:
        name = str(source["source_model"])
        key_rows.append(
            {
                "blind_id": _blind_id(lock_hash, str(block["physics_id"]), name),
                "physics_id": str(block["physics_id"]),
                **source,
            }
        )
    blind_by_curve = {
        str(row["curve_model"]): str(row["blind_id"]) for row in key_rows
    }

    score_rows: list[dict[str, Any]] = []
    for history_id in range(profile.pairs_per_coupling):
        base = POC._exact_curves_for_pair(
            profile,
            coupling_seed=int(block["coupling_seed"]),
            pair_id=history_id,
            reservoir=reservoir,
            spatial_channel=channel,
            observables=observables,
        )
        curves = _augmented_curve_pairs(base, profile.suffix_lengths)
        for (curve_model, condition, suffix_length), (x_a, x_b) in curves.items():
            if curve_model not in source_by_curve:
                continue
            exact = float(np.mean((x_a - x_b) ** 2))
            common = {
                "blind_id": blind_by_curve[curve_model],
                "physics_id": str(block["physics_id"]),
                "history_id": int(history_id),
                "condition": condition,
                "suffix_length": int(suffix_length),
                "n_observables": int(len(observables)),
                "exact_signal_mse": exact,
            }
            score_rows.append(
                {
                    **common,
                    "shots_per_batch": INFINITY,
                    "measurement_seed": 0,
                    "cross_batch_signal_mse": exact,
                }
            )
            parts = (
                int(block["coupling_seed"]),
                int(history_id),
                int(source_codes[curve_model]),
                int(condition_codes[condition]),
                int(suffix_codes[int(suffix_length)]),
            )
            for shots in shot_tiers:
                for measurement_seed in measurement_seeds:
                    estimate, _ = POC._cross_batch_estimate(
                        x_a,
                        x_b,
                        tier=shots,
                        seed=measurement_seed,
                        parts=parts,
                    )
                    score_rows.append(
                        {
                            **common,
                            "shots_per_batch": str(shots),
                            "measurement_seed": int(measurement_seed),
                            "cross_batch_signal_mse": estimate,
                        }
                    )
    scores = pd.DataFrame(score_rows).sort_values(
        [
            "blind_id",
            "shots_per_batch",
            "measurement_seed",
            "condition",
            "suffix_length",
            "history_id",
        ],
        kind="mergesort",
    )
    key = pd.DataFrame(key_rows).sort_values("blind_id", kind="mergesort")
    if FORBIDDEN_CLASSIFIER_COLUMNS & set(scores):
        raise RuntimeError("blinded score table contains source labels")
    return scores.reset_index(drop=True), key.reset_index(drop=True)


def collect_split(
    config: dict[str, Any],
    lock_hash: str,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = config[f"{split}_blocks"]
    score_frames: list[pd.DataFrame] = []
    key_frames: list[pd.DataFrame] = []
    for index, block in enumerate(blocks, start=1):
        print(
            json.dumps(
                {
                    "event": "collect_block",
                    "split": split,
                    "index": index,
                    "total": len(blocks),
                    "physics_id": block["physics_id"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        scores, key = _collect_block(config, lock_hash, block)
        score_frames.append(scores)
        key_frames.append(key)
    return (
        pd.concat(score_frames, ignore_index=True),
        pd.concat(key_frames, ignore_index=True),
    )


def _fingerprint_rule(config: dict[str, Any]) -> FingerprintRule:
    raw = config["classifier"]
    return FingerprintRule(
        max_window=int(config["scope"]["max_window"]),
        baseline_floor=float(raw["baseline_floor"]),
        equivalence_ratio=float(raw["equivalence_ratio"]),
        liveness_ratio=float(raw["liveness_ratio"]),
        recurrent_tail_ratio=float(raw["recurrent_tail_ratio"]),
        reset_max_ratio=float(raw["reset_max_ratio"]),
        reset_contrast_ratio=float(raw["reset_contrast_ratio"]),
        tail_suffixes=tuple(int(x) for x in raw["tail_suffixes"]),
        confidence=float(raw.get("confidence", 0.95)),
    )


def _classify(
    scores: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    classifier = config["classifier"]
    return classify_blind(
        scores,
        _fingerprint_rule(config),
        n_boot=int(classifier["within_system_bootstrap"]),
        bootstrap_seed=int(classifier["bootstrap_seed"]),
    )


def _prediction_family(label: str) -> str:
    if str(label).startswith("FINITE_WINDOW_COMPATIBLE"):
        return "finite_window"
    if str(label) == "RECURRENT_RESET_SENSITIVE":
        return "recurrent_reset_sensitive"
    if str(label) == "RECURRENT_RESET_INSENSITIVE":
        return "recurrent_reset_insensitive"
    if str(label) == "INDETERMINATE_LOW_SIGNAL":
        return "low_signal"
    return "indeterminate"


def _metric_point(frame: pd.DataFrame) -> dict[str, float]:
    primary = frame[frame["primary_case"].astype(bool)]
    recurrent = primary[primary["expected_behavior"] == "recurrent_reset_sensitive"]
    bounded = primary[primary["expected_behavior"] == "finite_window"]
    sensitivity = (
        float((recurrent["predicted_family"] == "recurrent_reset_sensitive").mean())
        if len(recurrent)
        else float("nan")
    )
    specificity = (
        float((bounded["predicted_family"] == "finite_window").mean())
        if len(bounded)
        else float("nan")
    )
    balanced = float(0.5 * (sensitivity + specificity))
    abstained = primary["predicted_family"].isin(["low_signal", "indeterminate"])
    coverage = float((~abstained).mean())
    correct = (
        (
            (primary["expected_behavior"] == "recurrent_reset_sensitive")
            & (primary["predicted_family"] == "recurrent_reset_sensitive")
        )
        | (
            (primary["expected_behavior"] == "finite_window")
            & (primary["predicted_family"] == "finite_window")
        )
    )
    correct_rate = float(correct.mean())
    abstention_rate = float(abstained.mean())
    error_rate = float((~correct & ~abstained).mean())
    false_recurrent = (
        float(
            bounded["predicted_family"].isin(
                ["recurrent_reset_sensitive", "recurrent_reset_insensitive"]
            ).mean()
        )
        if len(bounded)
        else float("nan")
    )
    qelm = bounded[bounded["actual_window"].notna()]
    cutoff_ok = (
        (qelm["inferred_cutoff"].astype(float) - qelm["actual_window"].astype(float))
        .abs()
        .le(1.0)
    )
    cutoff_accuracy = float(cutoff_ok.mean()) if len(qelm) else float("nan")
    return {
        "balanced_accuracy": balanced,
        "recurrent_sensitivity": sensitivity,
        "finite_specificity": specificity,
        "coverage": coverage,
        "correct_rate": correct_rate,
        "error_rate": error_rate,
        "abstention_rate": abstention_rate,
        "false_recurrent_rate": false_recurrent,
        "cutoff_within_one": cutoff_accuracy,
    }


def _metric_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return auditable numerators and denominators for every primary rate."""
    primary = frame[frame["primary_case"].astype(bool)]
    recurrent = primary[
        primary["expected_behavior"] == "recurrent_reset_sensitive"
    ]
    bounded = primary[primary["expected_behavior"] == "finite_window"]
    predicted_recurrent = bounded["predicted_family"].isin(
        ["recurrent_reset_sensitive", "recurrent_reset_insensitive"]
    )
    abstained = primary["predicted_family"].isin(["low_signal", "indeterminate"])
    correct = (
        (
            (primary["expected_behavior"] == "recurrent_reset_sensitive")
            & (primary["predicted_family"] == "recurrent_reset_sensitive")
        )
        | (
            (primary["expected_behavior"] == "finite_window")
            & (primary["predicted_family"] == "finite_window")
        )
    )
    cutoff_ok = (
        (
            bounded["inferred_cutoff"].astype(float)
            - bounded["actual_window"].astype(float)
        )
        .abs()
        .le(1.0)
    )
    return {
        "recurrent_sensitivity_numerator": int(
            (recurrent["predicted_family"] == "recurrent_reset_sensitive").sum()
        ),
        "recurrent_sensitivity_denominator": int(len(recurrent)),
        "finite_specificity_numerator": int(
            (bounded["predicted_family"] == "finite_window").sum()
        ),
        "finite_specificity_denominator": int(len(bounded)),
        "coverage_numerator": int((~abstained).sum()),
        "coverage_denominator": int(len(primary)),
        "correct_numerator": int(correct.sum()),
        "correct_denominator": int(len(primary)),
        "error_numerator": int((~correct & ~abstained).sum()),
        "error_denominator": int(len(primary)),
        "abstention_numerator": int(abstained.sum()),
        "abstention_denominator": int(len(primary)),
        "false_recurrent_numerator": int(predicted_recurrent.sum()),
        "false_recurrent_denominator": int(len(bounded)),
        "cutoff_within_one_numerator": int(cutoff_ok.sum()),
        "cutoff_within_one_denominator": int(len(bounded)),
    }


def _strict_exact_agreement_stats(frame: pd.DataFrame) -> tuple[float, int, int]:
    abstention_labels = {"INDETERMINATE", "INDETERMINATE_LOW_SIGNAL"}
    agreed = (
        ~frame["classification"].isin(abstention_labels)
        & ~frame["exact_classification"].isin(abstention_labels)
        & (frame["classification"] == frame["exact_classification"])
    )
    denominator = int(len(frame))
    numerator = int(agreed.sum())
    point = float(numerator / denominator) if denominator else float("nan")
    return point, numerator, denominator


def _nested_exact_agreement_interval(
    frame: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Resample physics blocks, then finite-shot seeds, keeping cases paired."""
    physics_ids = np.asarray(sorted(frame["physics_id"].unique()), dtype=object)
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    for _ in range(int(n_boot)):
        selected_physics = rng.choice(
            physics_ids,
            size=len(physics_ids),
            replace=True,
        )
        chunks: list[pd.DataFrame] = []
        for physics_id in selected_physics:
            physics = frame[frame["physics_id"] == physics_id]
            seeds = np.asarray(
                sorted(physics["measurement_seed"].unique()),
                dtype=int,
            )
            selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            for measurement_seed in selected_seeds:
                chunks.append(
                    physics[physics["measurement_seed"] == measurement_seed]
                )
        point, _, _ = _strict_exact_agreement_stats(
            pd.concat(chunks, ignore_index=True)
        )
        values.append(point)
    return tuple(float(x) for x in np.nanpercentile(values, [2.5, 97.5]))


def _cluster_metric_intervals(
    frame: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    physics_ids = np.asarray(sorted(frame["physics_id"].unique()), dtype=object)
    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {}
    for _ in range(int(n_boot)):
        selected = rng.choice(physics_ids, size=len(physics_ids), replace=True)
        chunks = []
        for draw, physics_id in enumerate(selected):
            chunk = frame[frame["physics_id"] == physics_id].copy()
            chunk["_cluster_draw"] = draw
            chunks.append(chunk)
        point = _metric_point(pd.concat(chunks, ignore_index=True))
        for name, value in point.items():
            samples.setdefault(name, []).append(float(value))
    return {
        name: tuple(float(x) for x in np.nanpercentile(values, [2.5, 97.5]))
        for name, values in samples.items()
    }


def _nested_metric_intervals(
    frame: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Bootstrap physics blocks, then shot realizations within each draw."""
    physics_ids = np.asarray(sorted(frame["physics_id"].unique()), dtype=object)
    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {}
    for _ in range(int(n_boot)):
        selected_physics = rng.choice(
            physics_ids,
            size=len(physics_ids),
            replace=True,
        )
        chunks: list[pd.DataFrame] = []
        for physics_draw, physics_id in enumerate(selected_physics):
            physics = frame[frame["physics_id"] == physics_id]
            seeds = np.asarray(
                sorted(physics["measurement_seed"].unique()),
                dtype=int,
            )
            selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            for seed_draw, measurement_seed in enumerate(selected_seeds):
                chunk = physics[
                    physics["measurement_seed"] == measurement_seed
                ].copy()
                chunk["_physics_draw"] = physics_draw
                chunk["_seed_draw"] = seed_draw
                chunks.append(chunk)
        point = _metric_point(pd.concat(chunks, ignore_index=True))
        for name, value in point.items():
            samples.setdefault(name, []).append(float(value))
    return {
        name: tuple(float(x) for x in np.nanpercentile(values, [2.5, 97.5]))
        for name, values in samples.items()
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    key: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    merged = predictions.merge(key, on=["blind_id", "physics_id"], validate="many_to_one")
    merged["predicted_family"] = merged["classification"].map(_prediction_family)
    rows: list[dict[str, Any]] = []
    outer_boot = int(config["classifier"]["outer_bootstrap"])
    for (shots, measurement_seed), group in merged.groupby(
        ["shots_per_batch", "measurement_seed"], sort=True
    ):
        point = _metric_point(group)
        counts = _metric_counts(group)
        intervals = _cluster_metric_intervals(
            group,
            n_boot=outer_boot,
            seed=int(config["classifier"]["outer_bootstrap_seed"])
            + int(measurement_seed),
        )
        row: dict[str, Any] = {
            "shots_per_batch": str(shots),
            "measurement_seed": str(int(measurement_seed)),
            "n_physics": int(group["physics_id"].nunique()),
        }
        for name, value in point.items():
            row[name] = value
            row[f"{name}_ci_low"] = intervals[name][0]
            row[f"{name}_ci_high"] = intervals[name][1]
        row.update(counts)
        rows.append(row)
    for shots, group in merged.groupby("shots_per_batch", sort=True):
        point = _metric_point(group)
        counts = _metric_counts(group)
        intervals = _nested_metric_intervals(
            group,
            n_boot=outer_boot,
            seed=int(config["classifier"]["outer_bootstrap_seed"]),
        )
        row = {
            "shots_per_batch": str(shots),
            "measurement_seed": "aggregate",
            "n_physics": int(group["physics_id"].nunique()),
        }
        for name, value in point.items():
            row[name] = value
            row[f"{name}_ci_low"] = intervals[name][0]
            row[f"{name}_ci_high"] = intervals[name][1]
        row.update(counts)
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values(
        ["shots_per_batch", "measurement_seed"], kind="mergesort"
    )

    exact = merged[merged["shots_per_batch"] == INFINITY][
        ["blind_id", "classification"]
    ].rename(columns={"classification": "exact_classification"})
    primary_label = str(config["measurement"]["primary_shots_per_batch"])
    finite = merged[merged["shots_per_batch"] == primary_label].merge(
        exact,
        on="blind_id",
        validate="many_to_one",
    )
    finite_primary = finite[finite["primary_case"].astype(bool)]
    agreement, agreement_numerator, agreement_denominator = (
        _strict_exact_agreement_stats(finite_primary)
    )
    agreement_ci = _nested_exact_agreement_interval(
        finite_primary,
        n_boot=outer_boot,
        seed=int(config["classifier"]["outer_bootstrap_seed"]) + 50_000,
    )

    fault = merged[~merged["primary_case"].astype(bool)].copy()
    fault_summary = (
        fault.groupby(
            ["shots_per_batch", "source_model", "expected_behavior", "classification"],
            sort=True,
            as_index=False,
        )
        .agg(
            fraction=("blind_id", "size"),
            n_physics=("physics_id", "nunique"),
        )
    )
    totals = fault_summary.groupby(
        ["shots_per_batch", "source_model"], sort=False
    )["fraction"].transform("sum")
    fault_summary["fraction"] = fault_summary["fraction"] / totals

    gates = config["acceptance_gates"]
    primary_metrics = metrics[
        (metrics["shots_per_batch"] == primary_label)
        & (metrics["measurement_seed"] == "aggregate")
    ]
    aggregate = {
        name: float(primary_metrics[name].mean())
        for name in (
            "balanced_accuracy",
            "recurrent_sensitivity",
            "finite_specificity",
            "coverage",
            "correct_rate",
            "error_rate",
            "abstention_rate",
            "false_recurrent_rate",
            "cutoff_within_one",
        )
    }
    for name in tuple(aggregate):
        aggregate[f"{name}_ci_low"] = float(
            primary_metrics[f"{name}_ci_low"].iloc[0]
        )
        aggregate[f"{name}_ci_high"] = float(
            primary_metrics[f"{name}_ci_high"].iloc[0]
        )
    for name in (
        "recurrent_sensitivity_numerator",
        "recurrent_sensitivity_denominator",
        "finite_specificity_numerator",
        "finite_specificity_denominator",
        "coverage_numerator",
        "coverage_denominator",
        "correct_numerator",
        "correct_denominator",
        "error_numerator",
        "error_denominator",
        "abstention_numerator",
        "abstention_denominator",
        "false_recurrent_numerator",
        "false_recurrent_denominator",
        "cutoff_within_one_numerator",
        "cutoff_within_one_denominator",
    ):
        aggregate[name] = int(primary_metrics[name].iloc[0])
    gate_results = {
        "balanced_accuracy": bool(
            aggregate["balanced_accuracy"] >= float(gates["balanced_accuracy"])
        ),
        "balanced_accuracy_cluster_lower": bool(
            aggregate["balanced_accuracy_ci_low"]
            >= float(gates["balanced_accuracy_cluster_lower"])
        ),
        "false_recurrent_rate": bool(
            aggregate["false_recurrent_rate"]
            <= float(gates["bounded_false_recurrent_max"])
        ),
        "cutoff_accuracy": bool(
            aggregate["cutoff_within_one"] >= float(gates["cutoff_within_one"])
        ),
        "exact_label_agreement": bool(
            agreement >= float(gates["exact_label_agreement"])
        ),
    }
    return metrics, fault_summary, {
        "primary_shots_per_batch": primary_label,
        "primary_metric_means": aggregate,
        "finite_vs_exact_label_agreement": agreement,
        "finite_vs_exact_label_agreement_numerator": agreement_numerator,
        "finite_vs_exact_label_agreement_denominator": agreement_denominator,
        "finite_vs_exact_label_agreement_ci_low": agreement_ci[0],
        "finite_vs_exact_label_agreement_ci_high": agreement_ci[1],
        "gate_results": gate_results,
        "all_primary_gates_pass": bool(all(gate_results.values())),
    }


def run_task_link(
    config: dict[str, Any],
    blocks: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    task = config["task_link"]
    taus = tuple(int(x) for x in task["taus"])
    max_delay = max(taus)
    rows: list[dict[str, Any]] = []
    for index, block in enumerate(blocks, start=1):
        n_qubits = int(block["n_qubits"])
        qrc_cfg = QRCConfig(
            dt=float(block["dt"]),
            h_in=float(block.get("h_in", 1.0)),
            n_virtual=1,
            n_qubits=n_qubits,
            J_s=float(block.get("J_s", 1.0)),
            h=float(block.get("h", 0.5)),
            gamma=float(block["gamma"]),
            dissipator=str(block["dissipator"]),
            quantize_levels=int(block.get("quantize_levels", 33)),
        )
        qelm_cfg = windowed_qelm_from_qrc(
            qrc_cfg,
            window_k=int(config["scope"]["max_window"]),
        )
        observables = readout.pauli_observables(
            n_qubits,
            axes=("x", "y", "z"),
            max_weight=int(block.get("observable_max_weight", 2)),
        )
        assert_matched(qrc_cfg, qelm_cfg, observables=observables)
        rng = np.random.default_rng(int(block["task_input_seed"]))
        inputs = rng.uniform(0.0, 1.0, size=int(task["rows"]))
        targets = task_families.build_family_targets(inputs, taus, (1,))
        bundles = {
            "qrc": build_features(
                qrc_cfg,
                inputs,
                int(block["coupling_seed"]),
                observables,
            ),
            "replay_qelm_d5": build_features(
                qelm_cfg,
                inputs,
                int(block["coupling_seed"]),
                observables,
            ),
        }
        valid = pipeline.stage_valid_indices(
            list(bundles.values()),
            targets,
            washout=int(task["washout"]),
            max_delay=max_delay,
        )
        splits = pipeline.make_splits(valid, tuple(task["split_fractions"]))
        for model, bundle in bundles.items():
            scored = pipeline.evaluate_feature_matrix(
                bundle.features,
                targets,
                splits,
                alphas=tuple(float(x) for x in task["alpha_grid"]),
                first_valid=bundle.first_valid,
            )
            for record in scored.to_dict(orient="records"):
                tau = int(str(record["target"]).split("_tau", 1)[1].split("_p", 1)[0])
                rows.append(
                    {
                        "physics_id": str(block["physics_id"]),
                        "model": model,
                        "tau": tau,
                        "capacity_test": float(record["capacity_test"]),
                        "alpha": float(record["alpha"]),
                        "feature_dim": int(record["feature_dim"]),
                    }
                )
        print(
            json.dumps(
                {
                    "event": "task_link_block",
                    "index": index,
                    "total": len(blocks),
                    "physics_id": block["physics_id"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    scores = pd.DataFrame(rows).sort_values(
        ["physics_id", "tau", "model"], kind="mergesort"
    )
    pivot = scores.pivot(index=["physics_id", "tau"], columns="model", values="capacity_test")
    pivot["delta_qrc_minus_qelm"] = pivot["qrc"] - pivot["replay_qelm_d5"]
    pivot = pivot.reset_index()
    short_taus = tuple(int(x) for x in task["short_taus"])
    long_taus = tuple(int(x) for x in task["long_taus"])
    system_rows: list[dict[str, Any]] = []
    for physics_id, group in pivot.groupby("physics_id", sort=True):
        short_delta = float(
            group[group["tau"].isin(short_taus)]["delta_qrc_minus_qelm"].mean()
        )
        long_delta = float(
            group[group["tau"].isin(long_taus)]["delta_qrc_minus_qelm"].mean()
        )
        system_rows.append(
            {
                "physics_id": physics_id,
                "short_delta": short_delta,
                "long_delta": long_delta,
                "crossover_shift": long_delta - short_delta,
                "positive_shift": bool(long_delta - short_delta > 0.0),
            }
        )
    systems = pd.DataFrame(system_rows)
    rng = np.random.default_rng(int(task["bootstrap_seed"]))
    shifts = systems["crossover_shift"].to_numpy(float)
    picks = rng.integers(0, len(shifts), size=(int(task["bootstrap"]), len(shifts)))
    boot_mean = shifts[picks].mean(axis=1)
    shift_ci = np.percentile(boot_mean, [2.5, 97.5])
    positive_fraction = float(systems["positive_shift"].mean())
    summary = {
        "n_physics": int(len(systems)),
        "mean_short_delta": float(systems["short_delta"].mean()),
        "mean_long_delta": float(systems["long_delta"].mean()),
        "mean_crossover_shift": float(shifts.mean()),
        "mean_crossover_shift_ci_low": float(shift_ci[0]),
        "mean_crossover_shift_ci_high": float(shift_ci[1]),
        "positive_shift_fraction": positive_fraction,
        "shift_gate": bool(
            shift_ci[0] > 0.0
            and positive_fraction
            >= float(config["acceptance_gates"]["positive_crossover_fraction"])
        ),
    }
    return scores, systems, summary


def validate_study_integrity(
    *,
    scores: pd.DataFrame,
    key: pd.DataFrame,
    predictions: pd.DataFrame,
    task_scores: pd.DataFrame,
    task_systems: pd.DataFrame,
    config: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    """Enforce the frozen row accounting before any source key is written."""
    blocks = config[f"{split}_blocks"]
    expected_physics = {str(block["physics_id"]) for block in blocks}
    histories = int(config["probe"]["histories_per_system"])
    suffixes = tuple(int(value) for value in config["probe"]["suffix_lengths"])
    conditions = tuple(str(value) for value in config["probe"]["conditions"])
    cases = int(config["device_panel"]["case_count_per_block"])
    shot_tiers = tuple(str(int(value)) for value in config["measurement"]["shots_per_batch"])
    measurement_seeds = tuple(
        int(value) for value in config["measurement"]["measurement_seeds"]
    )
    shot_replicates = 1 + len(shot_tiers) * len(measurement_seeds)
    expected_score_rows = (
        len(blocks)
        * cases
        * histories
        * len(suffixes)
        * len(conditions)
        * shot_replicates
    )
    expected_prediction_rows = len(blocks) * cases * shot_replicates
    expected_task_rows = len(blocks) * 2 * len(config["task_link"]["taus"])

    expected_shot_seed_pairs = {(INFINITY, 0)} | {
        (shots, seed) for shots in shot_tiers for seed in measurement_seeds
    }
    expected_history_ids = set(range(histories))
    expected_task_models = {"qrc", "replay_qelm_d5"}
    expected_task_taus = {int(value) for value in config["task_link"]["taus"]}
    expected_case_rows = {
        str(row["id"]): row for row in config["device_panel"]["cases"]
    }
    expected_case_ids = set(expected_case_rows)

    def id_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
        return set(
            zip(
                frame["blind_id"].astype(str),
                frame["physics_id"].astype(str),
            )
        )

    def shot_seed_pairs(frame: pd.DataFrame) -> set[tuple[str, int]]:
        return set(
            zip(
                frame["shots_per_batch"].astype(str),
                frame["measurement_seed"].astype(int),
            )
        )

    key_pairs = id_pairs(key)
    score_pairs = id_pairs(scores)
    prediction_pairs = id_pairs(predictions)
    expected_rows_per_case = histories * len(suffixes) * len(conditions) * shot_replicates

    case_metadata_matches = True
    n_qubits_by_physics = {
        str(block["physics_id"]): int(block["n_qubits"]) for block in blocks
    }
    for row in key.to_dict(orient="records"):
        source_model = str(row["source_model"])
        declared = expected_case_rows.get(source_model)
        if declared is None:
            case_metadata_matches = False
            break
        if (
            str(row["expected_behavior"]) != str(declared["expected_behavior"])
            or bool(row["primary_case"]) != bool(declared["primary_case"])
        ):
            case_metadata_matches = False
            break
        expected_window = declared.get("effective_window")
        if "effective_window_rule" in declared:
            expected_window = min(n_qubits_by_physics[str(row["physics_id"])], 3)
        actual_window = row["actual_window"]
        if expected_window is None:
            if not pd.isna(actual_window):
                case_metadata_matches = False
                break
        elif pd.isna(actual_window) or int(actual_window) != int(expected_window):
            case_metadata_matches = False
            break

    declared_dimensions = {
        int(n_qubits): int(dimension)
        for n_qubits, dimension in config["scope"]["observables"][
            "dimensions_by_n"
        ].items()
    }
    observable_dimensions_match = True
    for physics_id, group in scores.groupby("physics_id", sort=False):
        expected_dimension = declared_dimensions[
            n_qubits_by_physics[str(physics_id)]
        ]
        if set(group["n_observables"].astype(int)) != {expected_dimension}:
            observable_dimensions_match = False
            break

    checks = {
        "physics_ids": set(scores["physics_id"].astype(str)) == expected_physics,
        "key_physics_ids": set(key["physics_id"].astype(str)) == expected_physics,
        "score_rows": len(scores) == expected_score_rows,
        "key_rows": len(key) == len(blocks) * cases,
        "prediction_rows": len(predictions) == expected_prediction_rows,
        "task_score_rows": len(task_scores) == expected_task_rows,
        "task_system_rows": len(task_systems) == len(blocks),
        "score_key_id_pairs": score_pairs == key_pairs,
        "prediction_key_id_pairs": prediction_pairs == key_pairs,
        "score_rows_unique": not scores.duplicated(
            [
                "blind_id",
                "physics_id",
                "history_id",
                "condition",
                "suffix_length",
                "shots_per_batch",
                "measurement_seed",
            ]
        ).any(),
        "score_rows_per_case": bool(
            scores.groupby(["blind_id", "physics_id"], sort=False)
            .size()
            .eq(expected_rows_per_case)
            .all()
        ),
        "prediction_rows_per_case": bool(
            predictions.groupby(["blind_id", "physics_id"], sort=False)
            .size()
            .eq(shot_replicates)
            .all()
        ),
        "score_cases_per_physics": bool(
            scores.groupby("physics_id", sort=False)["blind_id"]
            .nunique()
            .eq(cases)
            .all()
        ),
        "prediction_cases_per_physics": bool(
            predictions.groupby("physics_id", sort=False)["blind_id"]
            .nunique()
            .eq(cases)
            .all()
        ),
        "key_cases_per_physics": bool(
            key.groupby("physics_id", sort=False)["source_model"]
            .agg(lambda values: set(values.astype(str)) == expected_case_ids)
            .all()
        ),
        "key_case_metadata": case_metadata_matches,
        "observable_dimensions": observable_dimensions_match,
        "conditions": set(scores["condition"].astype(str)) == set(conditions),
        "suffixes": set(scores["suffix_length"].astype(int)) == set(suffixes),
        "history_ids": set(scores["history_id"].astype(int))
        == expected_history_ids,
        "shot_tiers": set(scores["shots_per_batch"].astype(str))
        == {INFINITY, *shot_tiers},
        "score_shot_seed_pairs": shot_seed_pairs(scores)
        == expected_shot_seed_pairs,
        "prediction_shot_seed_pairs": shot_seed_pairs(predictions)
        == expected_shot_seed_pairs,
        "finite_measurement_seeds": set(
            scores.loc[
                scores["shots_per_batch"].astype(str) != INFINITY,
                "measurement_seed",
            ].astype(int)
        )
        == set(measurement_seeds),
        "exact_measurement_seed": set(
            scores.loc[
                scores["shots_per_batch"].astype(str) == INFINITY,
                "measurement_seed",
            ].astype(int)
        )
        == {0},
        "blind_ids_unique_in_key": bool(key["blind_id"].is_unique),
        "prediction_groups_unique": not predictions.duplicated(
            ["blind_id", "physics_id", "shots_per_batch", "measurement_seed"]
        ).any(),
        "task_score_physics_ids": set(task_scores["physics_id"].astype(str))
        == expected_physics,
        "task_system_physics_ids": set(task_systems["physics_id"].astype(str))
        == expected_physics,
        "task_system_physics_unique": bool(task_systems["physics_id"].is_unique),
        "task_score_grid": set(task_scores["model"].astype(str))
        == expected_task_models
        and set(task_scores["tau"].astype(int)) == expected_task_taus,
        "task_score_rows_unique": not task_scores.duplicated(
            ["physics_id", "model", "tau"]
        ).any(),
        "task_score_rows_per_physics": bool(
            task_scores.groupby("physics_id", sort=False)
            .size()
            .eq(len(expected_task_models) * len(expected_task_taus))
            .all()
        ),
        "finite_scores_present": bool(
            np.isfinite(scores["cross_batch_signal_mse"].to_numpy(float)).all()
        ),
        "finite_task_scores_present": bool(
            np.isfinite(task_scores["capacity_test"].to_numpy(float)).all()
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"frozen study integrity failure: {', '.join(failed)}")
    return {
        "pass": True,
        "checks": checks,
        "n_physics": len(blocks),
        "score_rows": int(len(scores)),
        "prediction_rows": int(len(predictions)),
        "task_score_rows": int(len(task_scores)),
        "expected_score_rows": int(expected_score_rows),
        "expected_prediction_rows": int(expected_prediction_rows),
        "expected_task_score_rows": int(expected_task_rows),
    }


def plot_heldout_summary(
    unblinded: pd.DataFrame,
    task_scores: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict[str, Any],
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fingerprint_figure": out_dir / "recurrence_fingerprint_heldout.png",
        "task_figure": out_dir / "task_reversal_heldout.png",
    }

    exact = unblinded[
        (unblinded["shots_per_batch"] == INFINITY)
        & unblinded["source_model"].isin(
            ["qrc", "replay_qelm_d5", "forced_reset_qrc_d3", "leaky_qelm_control"]
        )
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    label_map = {
        "qrc": "QRC",
        "replay_qelm_d5": "TD-QELM d=5",
        "forced_reset_qrc_d3": "forced-reset QRC",
        "leaky_qelm_control": "state-leaky control",
    }
    colors = {
        "qrc": "#b2182b",
        "replay_qelm_d5": "#2166ac",
        "forced_reset_qrc_d3": "#4d9221",
        "leaky_qelm_control": "#762a83",
    }
    for source, group in exact.groupby("source_model", sort=True):
        # The prediction table stores normalized point metrics, not every curve;
        # display tail/cutoff summaries as architecture-blind scalar evidence.
        x = np.arange(len(group))
        y = group["recurrent_tail_mean_ratio"].to_numpy(float)
        ax.scatter(
            x,
            y,
            s=24,
            alpha=0.7,
            color=colors[source],
            label=label_map[source],
        )
    ax.axhline(
        float(config["classifier"]["recurrent_tail_ratio"]),
        color="black",
        linestyle="--",
        linewidth=1,
        label="tail gate",
    )
    ax.set_xlabel("Anonymous held-out case (within source family)")
    ax.set_ylabel("Normalized same-suffix tail")
    ax.set_title("Blind held-out recurrence evidence")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(paths["fingerprint_figure"], dpi=220)
    fig.savefig(paths["fingerprint_figure"].with_suffix(".pdf"))
    plt.close(fig)

    pivot = task_scores.pivot_table(
        index="tau",
        columns="model",
        values="capacity_test",
        aggfunc=list,
    )
    taus = sorted(task_scores["tau"].unique())
    deltas: list[np.ndarray] = []
    for tau in taus:
        qrc = np.asarray(pivot.loc[tau, "qrc"], dtype=float)
        qelm = np.asarray(pivot.loc[tau, "replay_qelm_d5"], dtype=float)
        deltas.append(qrc - qelm)
    means = np.asarray([values.mean() for values in deltas])
    lows = np.asarray([np.percentile(values, 2.5) for values in deltas])
    highs = np.asarray([np.percentile(values, 97.5) for values in deltas])
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.plot(taus, means, marker="o", color="#6a3d9a", linewidth=2)
    ax.fill_between(taus, lows, highs, color="#cab2d6", alpha=0.5)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(5, color="#555555", linestyle="--", linewidth=1)
    ax.text(5.15, ax.get_ylim()[1] * 0.88, "public window bound", fontsize=8)
    ax.set_xlabel("Target delay")
    ax.set_ylabel(r"Test-capacity difference $C_{QRC}-C_{QELM}$")
    ax.set_title("Held-out task reversal across matched substrates")
    ax.grid(alpha=0.2)
    fig.savefig(paths["task_figure"], dpi=220)
    fig.savefig(paths["task_figure"].with_suffix(".pdf"))
    plt.close(fig)
    return paths


def _expected_row_accounting(
    config: dict[str, Any],
    split: str,
) -> dict[str, int]:
    blocks = config[f"{split}_blocks"]
    cases = int(config["device_panel"]["case_count_per_block"])
    histories = int(config["probe"]["histories_per_system"])
    suffixes = len(config["probe"]["suffix_lengths"])
    conditions = len(config["probe"]["conditions"])
    shot_replicates = 1 + len(config["measurement"]["shots_per_batch"]) * len(
        config["measurement"]["measurement_seeds"]
    )
    return {
        "scores": len(blocks)
        * cases
        * histories
        * suffixes
        * conditions
        * shot_replicates,
        "key": len(blocks) * cases,
        "predictions": len(blocks) * cases * shot_replicates,
        "task_scores": len(blocks) * 2 * len(config["task_link"]["taus"]),
        "task_systems": len(blocks),
    }


def _write_failure_outputs(
    *,
    out_dir: Path,
    split: str,
    config_path: Path,
    config: dict[str, Any],
    lock_hash: str,
    opening_manifest_path: Path,
    allow_dirty: bool,
    failure_stage: str,
    error: Exception,
    scores: pd.DataFrame | None,
    key: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    task_scores: pd.DataFrame | None,
    task_systems: pd.DataFrame | None,
    task_summary: dict[str, Any] | None,
) -> dict[str, Path]:
    """Close an opened split as NO-GO while preserving all available evidence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "opening_manifest": opening_manifest_path,
        "failure_ledger": out_dir / "recurrence_fingerprint_failures.csv",
        "result": out_dir / "recurrence_fingerprint_study.json",
    }
    frames = {
        "scores": scores,
        "key": key,
        "predictions": predictions,
        "task_scores": task_scores,
        "task_systems": task_systems,
    }
    frame_paths = {
        "scores": out_dir / "fingerprint_scores_blinded.csv",
        "key": out_dir / "fingerprint_unblinding_key_failure_closed.csv",
        "predictions": out_dir / "fingerprint_predictions_blinded.csv",
        "task_scores": out_dir / "task_link_scores.csv",
        "task_systems": out_dir / "task_link_systems.csv",
    }
    for name, frame in frames.items():
        if frame is not None:
            frame.to_csv(frame_paths[name], index=False, float_format="%.17g")
            paths[name] = frame_paths[name]

    expected = _expected_row_accounting(config, split)
    observed = {
        name: (0 if frame is None else int(len(frame)))
        for name, frame in frames.items()
    }
    ledger_rows: list[dict[str, Any]] = [
        {
            "kind": "exception",
            "stage": failure_stage,
            "table": "",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "expected_rows": "",
            "observed_rows": "",
            "missing_rows": "",
            "excess_rows": "",
        }
    ]
    for table, expected_rows in expected.items():
        observed_rows = observed[table]
        ledger_rows.append(
            {
                "kind": "row_accounting",
                "stage": failure_stage,
                "table": table,
                "error_type": "",
                "error_message": "",
                "expected_rows": expected_rows,
                "observed_rows": observed_rows,
                "missing_rows": max(expected_rows - observed_rows, 0),
                "excess_rows": max(observed_rows - expected_rows, 0),
            }
        )
    pd.DataFrame(ledger_rows).to_csv(paths["failure_ledger"], index=False)

    opening = json.loads(opening_manifest_path.read_text(encoding="utf-8"))
    opening_payload = opening["core"]["payload"]
    current_source_hashes = {
        name: manifests.sha256_file(path)
        for name, path in _study_source_files(config_path).items()
        if path.exists()
    }
    provenance_checks = {
        "config_hash_matches_lock": str(opening_payload["config_sha256"])
        == lock_hash,
        "config_unchanged_since_opening": manifests.sha256_file(config_path)
        == lock_hash,
        "source_hashes_unchanged_since_opening": opening_payload["source_sha256"]
        == current_source_hashes,
        "opened_before_collection_recorded": bool(
            opening_payload["opened_before_collection"]
        ),
    }
    accounting = {
        table: {
            "expected": expected[table],
            "observed": observed[table],
            "missing": max(expected[table] - observed[table], 0),
            "excess": max(observed[table] - expected[table], 0),
        }
        for table in expected
    }
    result = {
        "study": V1_FREEZE_NAME,
        "split": split,
        "simulator_only": True,
        "architecture_blind": True,
        "execution_status": "FAILED_CLOSED",
        "overall_verdict": "NO_GO",
        "config_sha256": lock_hash,
        "opening_manifest_sha256": manifests.sha256_file(opening_manifest_path),
        "source_sha256": current_source_hashes,
        "failure": {
            "stage": failure_stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "policy": (
                "No imputation or silent exclusion; missing or nonfinite rows "
                "close the opened heldout study as NO_GO."
            ),
        },
        "row_accounting": accounting,
        "integrity": {
            "pass": False,
            "checks": {
                "complete_registered_grid": False,
                "opening_provenance": bool(all(provenance_checks.values())),
            },
            "opening_provenance_checks": provenance_checks,
        },
        "evaluation": {
            "all_primary_gates_pass": False,
            "gate_results": {},
            "not_evaluated_due_to_failure": True,
        },
        "task_link": {
            **(task_summary or {}),
            "shift_gate": False,
        },
    }
    paths["result"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_hashes = {
        path.name: manifests.sha256_file(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file()
        and path.name != "recurrence_fingerprint_study_manifest.json"
    }
    paths["manifest"] = manifests.write_manifest(
        out_dir,
        "recurrence_fingerprint_study",
        {
            "repo_root": ROOT,
            "study": V1_FREEZE_NAME,
            "split": split,
            "allow_dirty": bool(allow_dirty),
            "execution_status": "FAILED_CLOSED",
            "overall_verdict": "NO_GO",
            "config_sha256": lock_hash,
            "opening_manifest_sha256": manifests.sha256_file(
                opening_manifest_path
            ),
            "source_sha256": current_source_hashes,
            "provenance_checks": provenance_checks,
            "failure": result["failure"],
            "row_accounting": accounting,
            "artifact_sha256": artifact_hashes,
        },
    )
    return paths


def _write_outputs(
    *,
    out_dir: Path,
    split: str,
    config_path: Path,
    config: dict[str, Any],
    lock_hash: str,
    scores: pd.DataFrame,
    key: pd.DataFrame,
    predictions: pd.DataFrame,
    task_scores: pd.DataFrame,
    task_systems: pd.DataFrame,
    task_summary: dict[str, Any],
    integrity: dict[str, Any],
    opening_manifest_path: Path,
    allow_dirty: bool,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "scores_blinded": out_dir / "fingerprint_scores_blinded.csv",
        "predictions_blinded": out_dir / "fingerprint_predictions_blinded.csv",
        "unblinding_key": out_dir / "fingerprint_unblinding_key.csv",
        "predictions_unblinded": out_dir / "fingerprint_predictions_unblinded.csv",
        "metrics": out_dir / "fingerprint_metrics.csv",
        "fault_controls": out_dir / "fingerprint_fault_controls.csv",
        "task_scores": out_dir / "task_link_scores.csv",
        "task_systems": out_dir / "task_link_systems.csv",
        "result": out_dir / "recurrence_fingerprint_study.json",
        "opening_manifest": opening_manifest_path,
    }
    scores.to_csv(paths["scores_blinded"], index=False, float_format="%.17g")
    predictions.to_csv(
        paths["predictions_blinded"], index=False, float_format="%.17g"
    )
    prediction_sha_before_unblinding = manifests.sha256_file(
        paths["predictions_blinded"]
    )
    key_sha_before_unblinding = _sha256_bytes(
        key.to_csv(index=False, float_format="%.17g").encode("utf-8")
    )
    key.to_csv(paths["unblinding_key"], index=False, float_format="%.17g")
    unblinded = predictions.merge(
        key,
        on=["blind_id", "physics_id"],
        validate="many_to_one",
    )
    unblinded.to_csv(
        paths["predictions_unblinded"], index=False, float_format="%.17g"
    )
    metrics, faults, evaluation = evaluate_predictions(predictions, key, config)
    metrics.to_csv(paths["metrics"], index=False, float_format="%.17g")
    faults.to_csv(paths["fault_controls"], index=False, float_format="%.17g")
    task_scores.to_csv(paths["task_scores"], index=False, float_format="%.17g")
    task_systems.to_csv(paths["task_systems"], index=False, float_format="%.17g")
    figure_paths = plot_heldout_summary(
        unblinded,
        task_scores,
        metrics,
        config,
        out_dir,
    )
    paths.update(figure_paths)
    paths["fingerprint_figure_pdf"] = figure_paths["fingerprint_figure"].with_suffix(".pdf")
    paths["task_figure_pdf"] = figure_paths["task_figure"].with_suffix(".pdf")

    # Recheck after classification, unblinding, evaluation, and plotting so a
    # mid-run source edit cannot inherit the opening provenance seal or a GO.
    closing_provenance = _validate_opening_manifest(
        opening_manifest_path=opening_manifest_path,
        config_path=config_path,
        lock_hash=lock_hash,
    )
    integrity["closing_provenance"] = closing_provenance
    source_hashes = closing_provenance["source_sha256"]
    result = {
        "study": "recurrence_fingerprint_v1",
        "split": split,
        "execution_status": "COMPLETE",
        "simulator_only": True,
        "architecture_blind": True,
        "config_sha256": lock_hash,
        "predictions_sha256_before_unblinding": prediction_sha_before_unblinding,
        "unblinding_key_sha256_before_write": key_sha_before_unblinding,
        "shot_model": "independent exact-binomial Pauli sampling; two batches per arm",
        "reset_scope": "ideal commanded reset in simulator; no hardware reset-fidelity claim",
        "source_sha256": source_hashes,
        "opening_manifest_sha256": manifests.sha256_file(opening_manifest_path),
        "integrity": integrity,
        "evaluation": evaluation,
        "task_link": task_summary,
        "overall_verdict": (
            "GO"
            if split == "heldout"
            and integrity["pass"]
            and evaluation["all_primary_gates_pass"]
            and task_summary["shift_gate"]
            else ("NO_GO" if split == "heldout" else "DESIGN_ONLY")
        ),
    }
    paths["result"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_hashes = {
        name: manifests.sha256_file(path)
        for name, path in paths.items()
        if path.exists()
    }
    manifest_path = manifests.write_manifest(
        out_dir,
        "recurrence_fingerprint_study",
        {
            "repo_root": ROOT,
            "study": "recurrence_fingerprint_v1",
            "split": split,
            "execution_status": "COMPLETE",
            "allow_dirty": bool(allow_dirty),
            "config_sha256": lock_hash,
            "predictions_sha256_before_unblinding": prediction_sha_before_unblinding,
            "unblinding_key_sha256_before_write": key_sha_before_unblinding,
            "source_sha256": source_hashes,
            "opening_manifest_sha256": manifests.sha256_file(
                opening_manifest_path
            ),
            "artifact_sha256": artifact_hashes,
            "evaluation": evaluation,
            "integrity": integrity,
            "task_link": task_summary,
        },
    )
    paths["manifest"] = manifest_path
    return paths


def run_study(
    *,
    config_path: Path,
    split: str,
    out_dir: Path,
    allow_dirty: bool,
) -> dict[str, Path]:
    canonical_heldout = (ROOT / "results_recurrence_fingerprint_heldout").resolve()
    if split == "heldout" and out_dir.resolve() != canonical_heldout:
        raise SystemExit(
            "heldout is open-once and must use the canonical output directory: "
            f"{canonical_heldout}"
        )
    config, lock_hash = load_locked_config(config_path)
    if split == "heldout" and str(config["freeze_name"]) != V1_FREEZE_NAME:
        raise SystemExit(
            f"heldout requires the frozen {V1_FREEZE_NAME} configuration"
        )
    opening_manifest_path = _write_opening_manifest(
        out_dir=out_dir,
        split=split,
        config_path=config_path,
        lock_hash=lock_hash,
        allow_dirty=allow_dirty,
    )
    scores: pd.DataFrame | None = None
    key: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    task_scores: pd.DataFrame | None = None
    task_systems: pd.DataFrame | None = None
    task_summary: dict[str, Any] | None = None
    stage = "collection"
    try:
        scores, key = collect_split(config, lock_hash, split)
        stage = "classification"
        predictions = _classify(scores, config)
        stage = "task_link"
        task_scores, task_systems, task_summary = run_task_link(
            config,
            list(config[f"{split}_blocks"]),
        )
        stage = "opening_provenance"
        provenance_integrity = _validate_opening_manifest(
            opening_manifest_path=opening_manifest_path,
            config_path=config_path,
            lock_hash=lock_hash,
        )
        stage = "integrity"
        integrity = validate_study_integrity(
            scores=scores,
            key=key,
            predictions=predictions,
            task_scores=task_scores,
            task_systems=task_systems,
            config=config,
            split=split,
        )
        integrity["opening_provenance"] = provenance_integrity
        stage = "output_evaluation"
        return _write_outputs(
            out_dir=out_dir,
            split=split,
            config_path=config_path,
            config=config,
            lock_hash=lock_hash,
            scores=scores,
            key=key,
            predictions=predictions,
            task_scores=task_scores,
            task_systems=task_systems,
            task_summary=task_summary,
            integrity=integrity,
            opening_manifest_path=opening_manifest_path,
            allow_dirty=allow_dirty,
        )
    except Exception as error:
        return _write_failure_outputs(
            out_dir=out_dir,
            split=split,
            config_path=config_path,
            config=config,
            lock_hash=lock_hash,
            opening_manifest_path=opening_manifest_path,
            allow_dirty=allow_dirty,
            failure_stage=stage,
            error=error,
            scores=scores,
            key=key,
            predictions=predictions,
            task_scores=task_scores,
            task_systems=task_systems,
            task_summary=task_summary,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "data" / "frozen" / "recurrence_fingerprint_v1.json",
    )
    parser.add_argument("--split", choices=("design", "heldout"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--copy-paper-figures",
        action="store_true",
        help="Copy validated figures/tables into paper/figures after the run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dirty = manifests.require_clean_tree(bool(args.allow_dirty))
    out_dir = args.out or ROOT / f"results_recurrence_fingerprint_{args.split}"
    paths = run_study(
        config_path=args.config,
        split=args.split,
        out_dir=out_dir,
        allow_dirty=bool(args.allow_dirty),
    )
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    if args.copy_paper_figures:
        if args.split != "heldout":
            raise SystemExit(
                "paper figures may only be copied from the heldout split"
            )
        if result.get("execution_status") == "FAILED_CLOSED":
            raise SystemExit(
                "heldout failed closed; no paper figures were copied"
            )
        paper = ROOT / "paper" / "figures"
        paper.mkdir(parents=True, exist_ok=True)
        for key in (
            "fingerprint_figure",
            "fingerprint_figure_pdf",
            "task_figure",
            "task_figure_pdf",
            "metrics",
            "fault_controls",
            "task_systems",
        ):
            shutil.copy2(paths[key], paper / paths[key].name)
    print(
        json.dumps(
            {
                "status": "complete",
                "split": args.split,
                "output": str(out_dir),
                "dirty_tree_recorded": bool(dirty["preregistration_violated"]),
                "execution_status": result.get("execution_status", "COMPLETE"),
                "overall_verdict": result["overall_verdict"],
                "fingerprint_gates_pass": result["evaluation"].get(
                    "all_primary_gates_pass", False
                ),
                "task_link_gate_pass": result["task_link"].get(
                    "shift_gate", False
                ),
            },
            sort_keys=True,
        )
    )
    return 1 if result.get("execution_status") == "FAILED_CLOSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
