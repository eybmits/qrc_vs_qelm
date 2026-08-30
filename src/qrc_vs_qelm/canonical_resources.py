"""Deterministic resource contracts for the canonical emulation study.

The scientific tables are intentionally deterministic.  Actual wall-clock
time is therefore kept in a separate, explicitly non-decision-bearing
artifact, while state widths and measurement/depth proxies are derived from
the frozen profile.
"""

from __future__ import annotations

from hashlib import sha256
from math import comb
from typing import Any, Iterable, Mapping


def pauli_observable_names(
    n_qubits: int,
    *,
    axes: Iterable[str],
    max_weight: int,
) -> tuple[str, ...]:
    """Return the repository's canonical single- and same-axis two-site names."""

    n = int(n_qubits)
    labels = tuple(str(axis) for axis in axes)
    names = [f"{axis}{site}" for axis in labels for site in range(n)]
    if int(max_weight) >= 2:
        names.extend(
            f"{axis}{first}{axis}{second}"
            for axis in labels
            for first in range(n)
            for second in range(first + 1, n)
        )
    return tuple(names)


def observable_names_sha256(names: Iterable[str]) -> str:
    """Hash an ordered observable-name contract without storing matrices."""

    payload = "\n".join(str(name) for name in names).encode("utf-8")
    return sha256(payload).hexdigest()


def expected_observable_contract(
    n_qubits: int,
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a count and identity hash from a frozen observable specification."""

    axes = tuple(str(axis) for axis in specification["axes"])
    max_weight = int(specification["max_weight"])
    names = pauli_observable_names(
        int(n_qubits),
        axes=axes,
        max_weight=max_weight,
    )
    expected_count = len(axes) * int(n_qubits)
    if max_weight >= 2:
        expected_count += len(axes) * comb(int(n_qubits), 2)
    if len(names) != expected_count:
        raise AssertionError("observable contract count is internally inconsistent")
    return {
        "count": int(expected_count),
        "names_sha256": observable_names_sha256(names),
        "axes": list(axes),
        "max_weight": max_weight,
    }


def canonical_resource_report(
    profile: Mapping[str, Any],
    *,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    """Build the registered resource report for one completed study run."""

    teacher_contracts = {
        str(int(n_qubits)): expected_observable_contract(
            int(n_qubits),
            profile["teacher"]["target_observables"],
        )
        for n_qubits in profile["teacher_n_qubits"]
    }
    student_n = int(profile["student"]["n_qubits"])
    student_contract = expected_observable_contract(
        student_n,
        profile["student"]["observables"],
    )
    student_axes = len(profile["student"]["observables"]["axes"])
    teacher_axes = len(profile["teacher"]["target_observables"]["axes"])
    budgets = [int(value) for value in profile["snapshot_budgets"]]
    return {
        "schema_version": 1,
        "study": str(profile["study"]),
        "profile": str(profile["profile"]),
        "decision_bearing": False,
        "runtime": {
            "wall_clock_seconds": float(wall_clock_seconds),
            "scope": "run_canonical_study after profile validation through table construction",
            "note": "machine-dependent descriptive timing; excluded from duplicate science hashes",
        },
        "state_width": {
            "teacher_density_matrix_complex_entries": {
                str(int(n_qubits)): int(4 ** int(n_qubits))
                for n_qubits in profile["teacher_n_qubits"]
            },
            "student_statevector_complex_amplitudes": int(2**student_n),
            "student_qubits": student_n,
        },
        "circuit_depth_proxies": {
            "teacher_recurrent_updates_per_new_input": 1,
            "student_parallel_encoding_layers": 1,
            "student_evolution_segments": int(profile["max_snapshots"]),
            "student_selected_snapshot_checkpoints": budgets,
            "note": "algorithmic segments, not transpiled hardware gate depth",
        },
        "measurement_setting_proxies": {
            "definition": (
                "one global tensor-product Pauli basis per axis and snapshot; "
                "same-axis single- and two-site observables are grouped"
            ),
            "teacher_target_observables": teacher_contracts,
            "teacher_global_basis_settings_per_time": int(teacher_axes),
            "student_observables": student_contract,
            "student_features_per_budget": {
                str(budget): int(budget * student_contract["count"])
                for budget in budgets
            },
            "student_global_basis_settings_per_budget": {
                str(budget): int(budget * student_axes) for budget in budgets
            },
            "expectation_mode": "exact statevector/density-matrix expectations",
            "finite_shots_used": False,
        },
        "panel_dimensions": {
            "windows": [int(value) for value in profile["windows"]],
            "suffix_groups": int(profile["suffix_groups"]),
            "prefix_branches_per_suffix": int(profile["prefix_branches"]),
            "prefix_burnin": int(profile["prefix_burnin"]),
            "train_rows": int(profile["rows"]["train"]),
            "validation_rows": int(profile["rows"]["val"]),
            "test_rows": int(profile["rows"]["test"]),
        },
    }


__all__ = [
    "canonical_resource_report",
    "expected_observable_contract",
    "observable_names_sha256",
    "pauli_observable_names",
]
