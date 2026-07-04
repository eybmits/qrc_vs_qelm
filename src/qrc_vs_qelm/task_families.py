"""Benchmark target families built on the core task generators.

Targets in this module are defined from the caller-provided input stream.  If a
feature builder quantizes its own drive, pass the original unquantized stream
here so every model is evaluated against the same target.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from numpy.polynomial import legendre as np_legendre

from qrc_vs_qelm.core import tasks as core_tasks


def _as_1d_float(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return arr


def _nonnegative_int(name: str, value: int) -> int:
    checked = int(value)
    if checked != value or checked < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return checked


def legendre_target(u: np.ndarray, tau: int, degree: int) -> np.ndarray:
    """Return ``y[t] = P_degree(2*u[t-tau] - 1)`` with leading NaNs."""

    values = _as_1d_float("u", u)
    tau = _nonnegative_int("tau", tau)
    degree = _nonnegative_int("degree", degree)

    y = np.full(values.shape[0], np.nan, dtype=np.float64)
    if tau >= values.shape[0]:
        return y

    coeffs = np.zeros(degree + 1, dtype=np.float64)
    coeffs[degree] = 1.0
    x = 2.0 * values[: values.shape[0] - tau] - 1.0
    y[tau:] = np_legendre.legval(x, coeffs)
    return y


def product_target(
    u: np.ndarray,
    tau1: int,
    p1: int,
    tau2: int,
    p2: int,
) -> np.ndarray:
    """Return the product of two delayed Legendre targets."""

    left = legendre_target(u, tau1, p1)
    right = legendre_target(u, tau2, p2)
    return left * right


def build_family_targets(
    u: np.ndarray,
    taus: Iterable[int],
    degrees: Iterable[int],
) -> dict[str, np.ndarray]:
    """Build all single-Legendre targets keyed as ``leg_tau{tau}_p{degree}``."""

    out: dict[str, np.ndarray] = {}
    for tau in taus:
        for degree in degrees:
            tau_i = _nonnegative_int("tau", tau)
            degree_i = _nonnegative_int("degree", degree)
            out[f"leg_tau{tau_i}_p{degree_i}"] = legendre_target(
                u,
                tau=tau_i,
                degree=degree_i,
            )
    return out


def _parse_product_spec(spec: Any) -> tuple[int, int, int, int]:
    if isinstance(spec, Mapping):
        if "tau" in spec:
            tau1 = tau2 = spec["tau"]
        else:
            tau1 = spec["tau1"]
            tau2 = spec["tau2"]
        return (
            _nonnegative_int("tau1", tau1),
            _nonnegative_int("p1", spec["p1"]),
            _nonnegative_int("tau2", tau2),
            _nonnegative_int("p2", spec["p2"]),
        )

    values = tuple(spec)
    if len(values) == 3:
        tau, p1, p2 = values
        return (
            _nonnegative_int("tau", tau),
            _nonnegative_int("p1", p1),
            _nonnegative_int("tau", tau),
            _nonnegative_int("p2", p2),
        )
    if len(values) == 4:
        tau1, p1, tau2, p2 = values
        return (
            _nonnegative_int("tau1", tau1),
            _nonnegative_int("p1", p1),
            _nonnegative_int("tau2", tau2),
            _nonnegative_int("p2", p2),
        )
    raise ValueError(
        "product specs must be (tau, p1, p2), (tau1, p1, tau2, p2), "
        "or a mapping with tau/tau1/tau2 and p1/p2"
    )


def _product_key(tau1: int, p1: int, tau2: int, p2: int) -> str:
    if tau1 == tau2:
        return f"prod_tau{tau1}_p{p1}{p2}"
    return f"prod_tau{tau1}_p{p1}_tau{tau2}_p{p2}"


def build_product_targets(
    u: np.ndarray,
    specs: Iterable[Any],
) -> dict[str, np.ndarray]:
    """Build product targets from deterministic tuple or mapping specs.

    A three-tuple ``(tau, p1, p2)`` uses the requested protocol key
    ``prod_tau{tau}_p{p1}{p2}``.  Four-tuples keep both delays in the key to
    avoid collisions.
    """

    out: dict[str, np.ndarray] = {}
    for spec in specs:
        tau1, p1, tau2, p2 = _parse_product_spec(spec)
        out[_product_key(tau1, p1, tau2, p2)] = product_target(
            u,
            tau1=tau1,
            p1=p1,
            tau2=tau2,
            p2=p2,
        )
    return out


def narma_targets(u: np.ndarray) -> dict[str, np.ndarray]:
    """Return NARMA-2/5/10 targets using the core NARMA recursion."""

    values = _as_1d_float("u", u)
    return {
        f"narma{order}": core_tasks.narma_target(values, order=order)
        for order in (2, 5, 10)
    }


def stm_targets(u: np.ndarray, taus: Iterable[int]) -> dict[str, np.ndarray]:
    """Return short-term memory targets keyed as ``stm_tau{tau}``."""

    values = _as_1d_float("u", u)
    return {
        f"stm_tau{tau_i}": core_tasks.delayed_target(values, tau_i)
        for tau_i in (_nonnegative_int("tau", tau) for tau in taus)
    }


def parity_targets(bits: np.ndarray, taus: Iterable[int]) -> dict[str, np.ndarray]:
    """Return temporal parity targets for a binary input stream."""

    values = _as_1d_float("bits", bits)
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("bits must contain only 0/1 values")
    return {
        f"parity_tau{tau_i}": core_tasks.parity_target(values, tau_i)
        for tau_i in (_nonnegative_int("tau", tau) for tau in taus)
    }


def load_santa_fe(path: str | Path = "data/santa_fe.npy") -> np.ndarray | None:
    """Load a cached Santa Fe laser series, returning ``None`` if absent."""

    file_path = Path(path)
    if not file_path.exists():
        warnings.warn(
            f"Santa Fe cache file not found: {file_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return np.asarray(np.load(file_path), dtype=np.float64).reshape(-1)
