from __future__ import annotations

import numpy as np
import pytest

from analyze_channel_spectrum import (
    channel_metrics,
    run_spectral_diagnostic,
    summarize_spectrum,
    traceless_pauli_basis,
)


def test_traceless_pauli_basis_is_complete_and_orthonormal() -> None:
    for n_qubits in (1, 2, 3):
        basis = traceless_pauli_basis(n_qubits)
        dim = 2**n_qubits
        assert basis.shape == (dim**2, dim**2 - 1)
        np.testing.assert_allclose(
            basis.conj().T @ basis,
            np.eye(dim**2 - 1),
            atol=1e-13,
        )


def test_identity_channel_has_only_peripheral_traceless_modes() -> None:
    basis = traceless_pauli_basis(2)
    metrics = channel_metrics(np.eye(16, dtype=complex), basis)
    assert metrics["spectral_radius_traceless"] == 1.0
    assert metrics["spectral_gap_one_step"] == 0.0
    assert metrics["peripheral_mode_count"] == 15
    assert metrics["slow_mode_count_ge_0p99"] == 15
    assert metrics["trace_preservation_error"] == 0.0


def test_smoke_spectrum_is_complete_and_summarizes() -> None:
    results = run_spectral_diagnostic(profile="smoke", input_value=0.5)
    assert len(results) == 3
    assert set(results["dissipator_family"]) == {
        "local_loss",
        "dephasing",
        "collective_loss",
    }
    assert np.all(results["spectral_radius_traceless"] <= 1.0 + 2e-9)
    summary = summarize_spectrum(results, profile="smoke")
    assert summary["counts"] == {"rows": 3, "roots": 1}
    assert len(summary["condition_summaries"]) == 3

    mutated = results.copy()
    mutated.loc[0, "reference_local_loss_gamma"] = 0.3
    with pytest.raises(ValueError, match="Cartesian profile"):
        summarize_spectrum(mutated, profile="smoke")
