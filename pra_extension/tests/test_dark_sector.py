from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "pra_extension" / "code"
SRC_ROOT = REPO_ROOT / "src"
for path in (CODE_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dark_sector import (  # noqa: E402
    addressable_x_pulse,
    asymmetric_singlet_initial_decay,
    collective_jumps,
    common_suffix_bound,
    evolve_pair,
    expectation,
    lindblad_protection_defect_upper_bound,
    negativity,
    robust_common_suffix_gap_bound,
    singlet_audit,
    singlet_ket,
    singlet_pauli_projector,
    singlet_projector,
    symmetric_two_qubit_terms,
    zero_state,
)
from qrc_vs_qelm.core import dissipators, liouvillian  # noqa: E402


def test_singlet_projector_is_entangled_and_has_exact_pauli_form() -> None:
    projector = singlet_projector()
    np.testing.assert_allclose(projector, singlet_pauli_projector(), atol=1e-14)
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-14)
    np.testing.assert_allclose(np.trace(projector), 1.0, atol=1e-14)
    np.testing.assert_allclose(negativity(projector), 0.5, atol=1e-14)


def test_symmetric_collective_qrc_protects_singlet_exactly() -> None:
    h_static, h_drive = symmetric_two_qubit_terms(0.73, 0.41, 0.62)
    jumps = collective_jumps(0.8)
    for suffix_input in (0.0, 0.17, 0.93, 1.0):
        audit = singlet_audit(h_static + suffix_input * h_drive, jumps)
        assert max(audit.to_dict().values()) < 2e-14


def test_symmetric_architecture_is_protected_but_unexcited() -> None:
    h_static, h_drive = symmetric_two_qubit_terms(-0.57, 0.39, 0.61)
    rho = zero_state()
    projector = singlet_projector()
    jumps = collective_jumps(0.8)
    for value in np.linspace(0.0, 1.0, 31):
        generator = liouvillian.lindbladian(h_static + value * h_drive, jumps)
        rho = liouvillian.evolve_step(generator, rho, 0.2)
        assert abs(expectation(rho, projector)) < 2e-14


def test_one_addressable_old_pulse_excites_a_known_protected_population() -> None:
    theta = 1.0
    pulsed = addressable_x_pulse(zero_state(), theta, site=0)
    expected = np.sin(theta / 2.0) ** 2 / 2.0
    np.testing.assert_allclose(
        expectation(pulsed, singlet_projector()), expected, atol=1e-14
    )


def test_common_suffix_error_floor_is_exact_for_every_suffix_length() -> None:
    theta = 1.0
    first = zero_state()
    second = addressable_x_pulse(first, theta, site=0)
    h_static, h_drive = symmetric_two_qubit_terms(0.83, 0.37, 0.64)
    rng = np.random.default_rng(28_082_802)
    records = evolve_pair(
        first,
        second,
        h_static,
        h_drive,
        collective_jumps(0.8),
        rng.uniform(0.0, 1.0, size=80),
        0.2,
    )
    gap = np.sin(theta / 2.0) ** 2 / 2.0
    np.testing.assert_allclose([row["population_gap"] for row in records], gap, atol=3e-14)
    np.testing.assert_allclose(
        [row["suffix_only_mse_floor"] for row in records], gap * gap / 4.0, atol=3e-15
    )


def test_balanced_two_history_bound_matches_direct_optimization() -> None:
    bound = common_suffix_bound(-0.3, 0.7)
    grid = np.linspace(-1.0, 1.0, 20_001)
    mse = 0.5 * ((grid + 0.3) ** 2 + (grid - 0.7) ** 2)
    worst = np.maximum(np.abs(grid + 0.3), np.abs(grid - 0.7))
    np.testing.assert_allclose(np.min(mse), bound.minimum_mean_squared_error, atol=1e-12)
    np.testing.assert_allclose(
        np.min(worst), bound.minimum_worst_case_absolute_error, atol=1e-12
    )


def test_local_loss_breaks_the_singlet_conservation_law() -> None:
    h_static, _ = symmetric_two_qubit_terms(0.0, 0.37, 0.0)
    audit = singlet_audit(h_static, dissipators.local_loss(2, 0.8))
    assert audit.adjoint_invariance_error > 0.1
    assert audit.jump_annihilation_error > 0.1


def test_asymmetric_collective_loss_has_exact_quadratic_initial_leakage() -> None:
    gamma = 0.8
    for epsilon in (0.01, 0.05, 0.2, 0.5):
        jump, rate = collective_jumps(gamma, epsilon)[0]
        direct = rate * float(np.linalg.norm(jump @ singlet_ket()) ** 2)
        np.testing.assert_allclose(
            asymmetric_singlet_initial_decay(gamma, epsilon), direct, atol=1e-15
        )
        # collective_loss normalizes c to ||c||^2=2
        c = np.array([1.0, 1.0 + epsilon])
        c *= np.sqrt(2.0 / np.sum(c * c))
        analytic = gamma * (c[1] - c[0]) ** 2 / 2.0
        np.testing.assert_allclose(direct, analytic, atol=1e-15)


def test_approximate_protection_certificate_bounds_every_sampled_suffix() -> None:
    theta = 1.0
    first = zero_state()
    second = addressable_x_pulse(first, theta, site=0)
    trace_distance = float(np.linalg.norm(first - second, ord="nuc"))
    gap0 = expectation(second, singlet_projector()) - expectation(
        first, singlet_projector()
    )
    h_static, h_drive = symmetric_two_qubit_terms(0.61, 0.37, 0.64)
    eta = 0.05
    jumps = collective_jumps(0.8, eta)
    defect = lindblad_protection_defect_upper_bound(
        liouvillian.lindbladian(h_static, jumps), singlet_projector(), 0.2
    )
    assert defect > 0
    records = evolve_pair(
        first,
        second,
        h_static,
        h_drive,
        jumps,
        np.random.default_rng(28_082_880).uniform(0.0, 1.0, size=80),
        0.2,
    )
    for row in records:
        certified = robust_common_suffix_gap_bound(
            gap0, defect, int(row["step"]), trace_distance
        )
        assert abs(float(row["population_gap"])) + 3e-13 >= certified


def test_exact_protection_has_zero_certificate_defect() -> None:
    h_static, _ = symmetric_two_qubit_terms(-0.43, 0.37, 0.64)
    defect = lindblad_protection_defect_upper_bound(
        liouvillian.lindbladian(h_static, collective_jumps(0.8)),
        singlet_projector(),
        0.2,
    )
    assert defect < 2e-14
    assert robust_common_suffix_gap_bound(0.1, defect, 1_000) > 0.0999999999
