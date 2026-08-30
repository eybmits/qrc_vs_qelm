from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "pra_extension" / "code"
SRC_ROOT = REPO_ROOT / "src"
for path in (CODE_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bernstein_qelm_validation import (  # noqa: E402
    bernstein_approximation,
    bernstein_lipschitz_bound,
    build_contracting_qrc,
    cartesian_grid,
    contraction_audit,
    distance_target,
    generic_bernstein_convergence,
    quantum_hamming_probabilities,
    qrc_bernstein_convergence,
    qrc_history_convergence,
    qrc_suffix_lipschitz_certificate,
    tensor_bernstein_probabilities,
)
from run_bernstein_qelm_validation import write_validation  # noqa: E402


def test_probabilities_partition_unity_and_match_statevector() -> None:
    points = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.13, 0.71], [0.5, 0.5]]
    )
    probabilities = tensor_bernstein_probabilities(points, r=5)
    assert np.min(probabilities) >= 0.0
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=2e-15)

    analytic = tensor_bernstein_probabilities([[0.17, 0.68]], r=3)[0]
    quantum = quantum_hamming_probabilities([0.17, 0.68], r=3)
    np.testing.assert_allclose(quantum, analytic, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(quantum.sum(), 1.0, atol=2e-15)


def test_constructive_readout_obeys_lipschitz_bound_and_converges() -> None:
    points = cartesian_grid(2, 51)
    reference = distance_target(points)
    errors = []
    for r in (1, 2, 4, 8, 16, 32):
        estimate = bernstein_approximation(points, r, distance_target)
        error = float(np.max(np.abs(estimate - reference)))
        errors.append(error)
        assert error <= bernstein_lipschitz_bound(1.0, d=2, r=r) + 2e-14
    assert errors[-1] < 0.25 * errors[0]


def test_existing_lindblad_qrc_has_uniform_contraction_certificate() -> None:
    model = build_contracting_qrc()
    audit = contraction_audit(model, gamma=4.0)
    assert audit["max_sampled_generator_log_norm"] <= -2.0 + 2e-13
    assert (
        audit["max_sampled_channel_trace_contraction"]
        <= audit["eta_certificate"] + 2e-13
    )

    certificate = qrc_suffix_lipschitz_certificate(
        d=2, gamma=4.0, dt=0.5, h_in=1.0
    )
    assert certificate["coordinate_constants_oldest_to_newest"] == pytest.approx(
        [np.exp(-1.0), 1.0]
    )
    assert certificate["euclidean_lipschitz_constant"] == pytest.approx(
        np.sqrt(1.0 + np.exp(-2.0))
    )


def test_qrc_replay_and_history_errors_obey_both_certificates() -> None:
    model = build_contracting_qrc()
    representation = qrc_bernstein_convergence(model)
    history = qrc_history_convergence(model)
    for row in representation:
        assert row["max_abs_error"] <= row["certified_max_error_bound"] + 2e-13
    for row in history:
        assert (
            row["max_abs_error"]
            <= row["certified_max_history_error_bound"] + 2e-13
        )
    assert representation[-1]["max_abs_error"] < representation[0]["max_abs_error"]
    assert history[-1]["max_abs_error"] < history[0]["max_abs_error"]


def test_artifact_writer_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    report_first = write_validation(first)
    report_second = write_validation(second)
    assert report_first["artifact_sha256"] == report_second["artifact_sha256"]
    assert sorted(path.name for path in first.iterdir()) == sorted(
        path.name for path in second.iterdir()
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    manifest = json.loads((first / "artifact_sha256.json").read_text())
    assert manifest == report_first["artifact_sha256"]
