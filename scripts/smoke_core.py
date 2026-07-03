from __future__ import annotations

import hashlib

import numpy as np

from qrc_vs_qelm.core import readout, reservoirs


def compute_smoke_features() -> np.ndarray:
    n_qubits = 4
    rng = np.random.default_rng(12345)
    J = reservoirs.random_couplings(n_qubits, J_s=1.0, rng=rng)
    reservoir = reservoirs.continuous_dissipation(
        n_qubits=n_qubits,
        J=J,
        h=0.5,
        gamma=0.1,
        dt=0.5,
        cache_propagators=False,
        quantize=None,
    )
    inputs = rng.uniform(0.0, 1.0, size=200)
    observables = readout.pauli_observables(n_qubits, max_weight=2)
    return reservoir.run(inputs, observables, washout=50)


def smoke_summary() -> tuple[tuple[int, int], str]:
    features = compute_smoke_features()
    digest = hashlib.sha256(features.tobytes()).hexdigest()[:16]
    return features.shape, digest


def main() -> None:
    shape, digest = smoke_summary()
    print(f"feature_matrix_shape={shape}")
    print(f"sha256_16={digest}")


if __name__ == "__main__":
    main()
