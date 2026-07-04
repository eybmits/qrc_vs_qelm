"""Classical matched-dimension control feature maps."""

from __future__ import annotations

import numpy as np


def _as_1d_float(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return arr


def _positive_int(name: str, value: int) -> int:
    checked = int(value)
    if checked != value or checked < 1:
        raise ValueError(f"{name} must be a positive integer")
    return checked


def _delay_window(u: np.ndarray, d: int) -> np.ndarray:
    values = _as_1d_float("u", u)
    d = _positive_int("d", d)
    X = np.zeros((values.shape[0], d), dtype=np.float64)
    for lag in range(min(d, values.shape[0])):
        X[lag:, lag] = values[: values.shape[0] - lag]
    return X


def linear_delay_features(u: np.ndarray, d: int, D: int) -> np.ndarray:
    """Return raw causal delay windows truncated or zero-padded to width ``D``.

    Column order is ``u[t]``, ``u[t-1]``, ..., ``u[t-d+1]``.  Unavailable
    leading lags are zero, and columns beyond the requested window are zero.
    """

    D = _positive_int("D", D)
    window = _delay_window(u, d)
    X = np.zeros((window.shape[0], D), dtype=np.float64)
    keep = min(D, window.shape[1])
    X[:, :keep] = window[:, :keep]
    return X


def poly_delay_features(u: np.ndarray, d: int, D: int) -> np.ndarray:
    """Return degree <= 2 delay-window features at exact width ``D``.

    Deterministic column order before truncation/padding is:
    linear terms ``x_i`` for ``i=0..d-1``; squares ``x_i^2`` for ``i=0..d-1``;
    then cross terms ``x_i*x_j`` for lexicographic pairs ``0 <= i < j < d``.
    Here ``x_i = u[t-i]`` with zero padding for unavailable leading lags.
    """

    D = _positive_int("D", D)
    window = _delay_window(u, d)
    X = np.zeros((window.shape[0], D), dtype=np.float64)

    col = 0
    for idx in range(window.shape[1]):
        if col >= D:
            return X
        X[:, col] = window[:, idx]
        col += 1
    for idx in range(window.shape[1]):
        if col >= D:
            return X
        X[:, col] = window[:, idx] ** 2
        col += 1
    for i in range(window.shape[1]):
        for j in range(i + 1, window.shape[1]):
            if col >= D:
                return X
            X[:, col] = window[:, i] * window[:, j]
            col += 1
    return X
