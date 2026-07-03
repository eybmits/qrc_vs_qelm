import numpy as np

from qrc_vs_qelm.core import operators as op
from qrc_vs_qelm.core import liouvillian as lio
from qrc_vs_qelm.core import dissipators as dsp
from qrc_vs_qelm.core import reservoirs as res


def random_dm(d, rng):
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def test_vec_unvec_roundtrip():
    rng = np.random.default_rng(0)
    rho = random_dm(4, rng)
    assert np.allclose(lio.unvec(lio.vec(rho), 4), rho)


def test_commutator_super_matches_direct():
    rng = np.random.default_rng(1)
    H = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    H = H + H.conj().T
    rho = random_dm(4, rng)
    direct = -1j * (H @ rho - rho @ H)
    via = lio.unvec(lio.commutator_super(H) @ lio.vec(rho), 4)
    assert np.allclose(direct, via)


def test_dissipator_super_matches_direct():
    rng = np.random.default_rng(2)
    L = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    rho = random_dm(4, rng)
    LdL = L.conj().T @ L
    direct = L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
    via = lio.unvec(lio.dissipator_super(L) @ lio.vec(rho), 4)
    assert np.allclose(direct, via)


def test_trace_preservation_and_hermiticity():
    N = 2
    rng = np.random.default_rng(3)
    H = op.pauli_op("z", 0, N) + op.pauli_op("x", 1, N)
    jumps = [(op.sminus(i, N), 0.7) for i in range(N)]
    Ls = lio.lindbladian(H, jumps)
    P = lio.propagator(Ls, 0.4)
    rho0 = random_dm(2 ** N, rng)
    rho1 = lio.apply_propagator(P, rho0)
    assert np.isclose(np.trace(rho1).real, 1.0, atol=1e-9)
    assert np.allclose(rho1, rho1.conj().T, atol=1e-9)


def test_amplitude_damping_steady_state():
    N = 2
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    jumps = [(op.sminus(i, N), 1.0) for i in range(N)]
    ss = lio.steady_state(lio.lindbladian(H, jumps))
    target = np.zeros((2 ** N, 2 ** N), dtype=complex)
    target[0, 0] = 1.0
    assert np.allclose(ss, target, atol=1e-6)


def test_local_loss_evolution_preserves_positivity_n3():
    N = 3
    rng = np.random.default_rng(4)
    J = res.random_couplings(N, J_s=0.7, rng=rng)
    H = res.ising_xx_hamiltonian(J, h=0.3, n_qubits=N)
    jumps = dsp.local_loss(N, gamma=0.4)
    Ls = lio.lindbladian(H, jumps)
    rho0 = random_dm(2 ** N, rng)

    for t in (0.0, 0.25, 1.0):
        P = lio.propagator(Ls, t)
        rho_out = lio.apply_propagator(P, rho0)
        rho_herm = 0.5 * (rho_out + rho_out.conj().T)
        assert np.min(np.linalg.eigvalsh(rho_herm)) >= -1e-10
        assert np.isclose(np.trace(rho_out), 1.0, atol=1e-9)
