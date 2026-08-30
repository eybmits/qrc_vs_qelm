"""Mid-drive channel-spectrum diagnostic for the PRA mechanism extension.

This analysis is deliberately narrower than a joint-spectral-radius or
arbitrary-input contraction theorem.  For every fresh root and matched
dissipator cell it restricts the one-step channel at a declared representative
input to the Hilbert--Schmidt-orthonormal traceless Pauli space.  The resulting
spectral radius tests for slow or peripheral operator modes.  It must be read
together with the sampled state-memory and observable-visibility diagnostics:
a slow mode need not be excited by the input history or seen by the measured
Pauli panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from observable_memory import _step_propagator, build_reservoir, default_config
from qrc_vs_qelm.core import liouvillian, operators


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def traceless_pauli_basis(n_qubits: int) -> np.ndarray:
    """Return vectorized nonidentity Pauli strings as orthonormal columns."""

    n_qubits = int(n_qubits)
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    dim = 2**n_qubits
    columns = [
        liouvillian.vec(operators.pauli_string(labels, n_qubits)) / np.sqrt(dim)
        for labels in product("ixyz", repeat=n_qubits)
        if any(label != "i" for label in labels)
    ]
    basis = np.stack(columns, axis=1)
    gram_error = float(
        np.max(np.abs(basis.conj().T @ basis - np.eye(basis.shape[1])))
    )
    if gram_error > 2e-13:
        raise RuntimeError(f"Pauli-basis orthogonality failed: {gram_error:.3e}")
    return basis


def channel_metrics(
    transfer: np.ndarray,
    basis: np.ndarray,
    *,
    peripheral_tolerance: float = 1e-8,
    slow_threshold: float = 0.99,
) -> dict[str, float | int]:
    """Diagnose a trace-preserving channel on the traceless Pauli subspace."""

    transfer = np.asarray(transfer, dtype=complex)
    basis = np.asarray(basis, dtype=complex)
    if transfer.shape[0] != transfer.shape[1] or transfer.shape[0] != basis.shape[0]:
        raise ValueError("transfer and Pauli-basis dimensions do not agree")
    restricted_complex = basis.conj().T @ transfer @ basis
    maximum_imaginary_entry = float(np.max(np.abs(restricted_complex.imag)))
    if maximum_imaginary_entry > 2e-10:
        raise RuntimeError("Hermiticity-preserving channel has a non-real Pauli block")
    restricted = restricted_complex.real
    eigenvalue_moduli = np.abs(np.linalg.eigvals(restricted))
    spectral_radius = float(np.max(eigenvalue_moduli))
    if spectral_radius > 1.0 + 2e-9:
        raise RuntimeError("CPTP-channel spectral radius exceeded one")
    singular_values = np.linalg.svd(restricted, compute_uv=False)

    dim = int(round(np.sqrt(transfer.shape[0])))
    trace_row = liouvillian.vec(np.eye(dim, dtype=complex)).conj()
    trace_preservation_error = float(
        np.max(np.abs(trace_row @ transfer - trace_row))
    )
    if trace_preservation_error > 2e-10:
        raise RuntimeError("channel trace-preservation audit failed")

    return {
        "spectral_radius_traceless": spectral_radius,
        "spectral_gap_one_step": float(max(0.0, 1.0 - spectral_radius)),
        "peripheral_mode_count": int(
            np.sum(eigenvalue_moduli >= 1.0 - peripheral_tolerance)
        ),
        "slow_mode_count_ge_0p99": int(np.sum(eigenvalue_moduli >= slow_threshold)),
        "largest_singular_value_traceless": float(singular_values[0]),
        "smallest_singular_value_traceless": float(singular_values[-1]),
        "maximum_imaginary_pauli_entry": maximum_imaginary_entry,
        "trace_preservation_error": trace_preservation_error,
    }


def run_spectral_diagnostic(
    *, profile: str = "full", input_value: float = 0.5
) -> pd.DataFrame:
    """Run the declared representative-input diagnostic over paired roots."""

    if not np.isfinite(input_value) or not 0.0 <= input_value <= 1.0:
        raise ValueError("input_value must be finite and lie in [0, 1]")
    config = default_config(profile)
    rows: list[dict[str, Any]] = []
    bases = {n: traceless_pauli_basis(n) for n in config.n_qubits}
    for root in config.roots:
        for n_qubits in config.n_qubits:
            coupling_seed = int(root.coupling_seeds[int(n_qubits)])
            for family, gamma in product(config.dissipator_families, config.gammas):
                model, budget = build_reservoir(
                    n_qubits=int(n_qubits),
                    family=str(family),
                    gamma=float(gamma),
                    coupling_seed=coupling_seed,
                    config=config,
                )
                transfer = _step_propagator(model, float(input_value))
                rows.append(
                    {
                        "root_id": root.root_id,
                        "coupling_seed": coupling_seed,
                        "teacher_n_qubits": int(n_qubits),
                        "dissipator_family": str(family),
                        "reference_local_loss_gamma": float(gamma),
                        "representative_input": float(input_value),
                        **budget,
                        **channel_metrics(transfer, bases[int(n_qubits)]),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        [
            "root_id",
            "teacher_n_qubits",
            "dissipator_family",
            "reference_local_loss_gamma",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_spectrum(
    results: pd.DataFrame, *, profile: str | None = None
) -> dict[str, Any]:
    required = {
        "root_id",
        "teacher_n_qubits",
        "dissipator_family",
        "reference_local_loss_gamma",
        "representative_input",
        "spectral_radius_traceless",
        "peripheral_mode_count",
        "slow_mode_count_ge_0p99",
        "trace_preservation_error",
        "maximum_imaginary_pauli_entry",
    }
    if missing := required - set(results):
        raise ValueError(f"spectral table missing columns: {sorted(missing)}")
    if results.duplicated(
        [
            "root_id",
            "teacher_n_qubits",
            "dissipator_family",
            "reference_local_loss_gamma",
        ]
    ).any():
        raise ValueError("duplicate spectral cells")
    numeric = [
        "teacher_n_qubits",
        "reference_local_loss_gamma",
        "representative_input",
        "spectral_radius_traceless",
        "peripheral_mode_count",
        "slow_mode_count_ge_0p99",
        "trace_preservation_error",
        "maximum_imaginary_pauli_entry",
    ]
    if not bool(np.isfinite(results[numeric].to_numpy(float)).all()):
        raise ValueError("spectral table contains nonfinite values")
    if results["representative_input"].nunique() != 1:
        raise ValueError("spectral table mixes representative inputs")
    if profile is not None:
        config = default_config(profile)
        expected = {
            (root.root_id, int(n_qubits), str(family), float(gamma))
            for root, n_qubits, family, gamma in product(
                config.roots,
                config.n_qubits,
                config.dissipator_families,
                config.gammas,
            )
        }
        observed = {
            (
                str(row.root_id),
                int(row.teacher_n_qubits),
                str(row.dissipator_family),
                float(row.reference_local_loss_gamma),
            )
            for row in results.itertuples(index=False)
        }
        if observed != expected or len(results) != len(expected):
            raise ValueError(
                f"spectral table differs from the exact {profile!r} Cartesian profile"
            )
    metrics: list[dict[str, Any]] = []
    keys = [
        "dissipator_family",
        "teacher_n_qubits",
        "reference_local_loss_gamma",
    ]
    for key, group in results.groupby(keys, sort=True):
        root_radii = group.groupby("root_id")["spectral_radius_traceless"].median()
        root_slow_counts = group.groupby("root_id")["slow_mode_count_ge_0p99"].median()
        metrics.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_roots": int(len(root_radii)),
                "median_spectral_radius_traceless": float(np.median(root_radii)),
                "minimum_spectral_radius_traceless": float(np.min(root_radii)),
                "maximum_spectral_radius_traceless": float(np.max(root_radii)),
                "median_slow_mode_count_ge_0p99": float(np.median(root_slow_counts)),
                "roots_with_peripheral_mode": int(
                    group.groupby("root_id")["peripheral_mode_count"].max().gt(0).sum()
                ),
            }
        )
    return {
        "schema_version": 1,
        "study": "representative_input_channel_spectrum",
        "status": "EXPLORATORY_MECHANISM_DIAGNOSTIC",
        "claim_boundary": (
            "one input-conditioned channel at the declared representative input; "
            "not a joint-spectral-radius or arbitrary-input contraction theorem"
        ),
        "counts": {
            "rows": int(len(results)),
            "roots": int(results["root_id"].nunique()),
        },
        "representative_input": float(results["representative_input"].iloc[0]),
        "numerical_audit": {
            "maximum_trace_preservation_error": float(
                results["trace_preservation_error"].max()
            ),
            "maximum_imaginary_pauli_entry": float(
                results["maximum_imaginary_pauli_entry"].max()
            ),
            "maximum_spectral_radius_excess": float(
                max(0.0, results["spectral_radius_traceless"].max() - 1.0)
            ),
        },
        "condition_summaries": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--input-value", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    results = run_spectral_diagnostic(
        profile=str(args.profile), input_value=float(args.input_value)
    )
    csv_path = output_dir / "channel_spectrum.csv"
    json_path = output_dir / "channel_spectrum_summary.json"
    results.to_csv(csv_path, index=False)
    summary = summarize_spectrum(results, profile=str(args.profile))
    summary["artifacts"] = {"channel_spectrum.csv": sha256(csv_path)}
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
