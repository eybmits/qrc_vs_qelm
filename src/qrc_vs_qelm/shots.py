"""Finite-shot noise utilities for Pauli expectation features."""

from __future__ import annotations

import numpy as np


def add_shot_noise(
    features: np.ndarray,
    shots: int | float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add binomial-normal shot noise to expectation features.

    Ported from
    ``/Users/superposition/Coding/qrc_dataset_profiler/src/qrc_dataset_profiler/reservoir.py``
    ``_add_shot_noise``.  The formula is valid for expectation values of
    ``+-1``-valued observables: ``std = sqrt((1 - f**2) / shots)``.  Inputs are
    clipped to ``[-1, 1]`` before noise is applied.  ``shots=None`` or
    ``shots=np.inf`` returns an unmodified float64 copy.
    """
    values = np.asarray(features, dtype=np.float64)
    if shots is None or np.isinf(shots):
        return values.copy()

    n_shots = max(int(shots), 1)
    clipped = np.clip(values, -1.0, 1.0)
    sigma = np.sqrt(np.maximum(1.0 - clipped**2, 0.0) / n_shots)
    noisy = clipped + rng.normal(0.0, sigma, clipped.shape)
    return np.clip(noisy, -1.0, 1.0).astype(np.float64)
