"""Resource-matching and feature-liveness guards."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from qrc_vs_qelm.core import diagnostics, liouvillian, readout, reservoirs
from qrc_vs_qelm.models.api import (
    FeatureBundle,
    QELMConfig,
    QRCConfig,
    _default_observables,
    _dissipator_jumps,
)


class MatchingError(ValueError):
    """Raised when a QRC/QELM pair violates a matched-resource contract."""


def _same_float(a: float, b: float) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=0.0))


def _n_observables(
    n_qubits: int,
    observables: Sequence[readout.Observable] | None,
) -> int:
    return len(_default_observables(n_qubits) if observables is None else observables)


def _shared_n_observables(
    qrc_cfg: QRCConfig,
    observables: Sequence[readout.Observable] | None,
) -> int:
    if observables is None:
        return _n_observables(qrc_cfg.n_qubits, None)

    obs = list(observables)
    if any(not isinstance(o, readout.Observable) for o in obs):
        raise MatchingError("observables must be a single shared Observable sequence")
    return len(obs)


def assert_matched(
    qrc_cfg: QRCConfig,
    qelm_cfg: QELMConfig,
    observables: Sequence[readout.Observable] | None = None,
) -> None:
    """Pre-flight identical config resources for one QRC/QELM pair.

    This check derives the readout width from one shared observable set. It is
    only a cheap config-level pre-flight; true built D-parity is certified by
    :func:`assert_bundles_matched`, which compares the actual feature bundles.
    """
    violations: list[str] = []
    if qrc_cfg.n_qubits != qelm_cfg.n_qubits:
        violations.append("n_qubits")
    if not _same_float(qrc_cfg.J_s, qelm_cfg.J_s):
        violations.append("J_s")
    if not _same_float(qrc_cfg.h, qelm_cfg.h):
        violations.append("h")
    if not _same_float(qrc_cfg.gamma, qelm_cfg.gamma):
        violations.append("gamma")
    if qrc_cfg.dissipator != qelm_cfg.dissipator:
        violations.append("dissipator")
    if qrc_cfg.quantize_levels != qelm_cfg.quantize_levels:
        violations.append("quantize_levels")
    if qrc_cfg.n_virtual != qelm_cfg.n_snapshots:
        violations.append("multiplexing")

    n_obs = _shared_n_observables(qrc_cfg, observables)
    qrc_dim = qrc_cfg.n_virtual * n_obs
    qelm_dim = qelm_cfg.n_snapshots * n_obs
    if qrc_dim != qelm_dim:
        violations.append("readout_dimension")

    if violations:
        names = ", ".join(dict.fromkeys(violations))
        raise MatchingError(f"matched-resource violations: {names}")


def assert_bundles_matched(
    qrc_bundle: FeatureBundle,
    qelm_bundle: FeatureBundle,
) -> None:
    """Assert actual built feature bundles have identical readout resources."""
    qrc_features = np.asarray(qrc_bundle.features)
    qelm_features = np.asarray(qelm_bundle.features)
    violations: list[str] = []

    if qrc_features.ndim != 2 or qelm_features.ndim != 2:
        raise MatchingError("feature bundles must contain two-dimensional features")
    if qrc_features.shape[1] != qelm_features.shape[1]:
        violations.append("readout_dimension")
    if int(qrc_bundle.n_obs) != int(qelm_bundle.n_obs):
        violations.append("n_obs")
    if int(qrc_bundle.n_multiplex) != int(qelm_bundle.n_multiplex):
        violations.append("n_multiplex")

    if violations:
        names = ", ".join(dict.fromkeys(violations))
        raise MatchingError(f"bundle-resource violations: {names}")


def physical_shot_budget(
    shots_per_expectation: int,
    n_timesteps: int,
    readout_dim: int,
) -> int:
    """Return the explicit physical budget ``S * T * D``."""
    return int(shots_per_expectation) * int(n_timesteps) * int(readout_dim)


def _participation_eff_rank(features: np.ndarray) -> float:
    X = np.asarray(features, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(Xc, compute_uv=False)
    denom = float(np.sum(singular_values**2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(singular_values) ** 2 / denom)


def effective_dimension(features: np.ndarray, *, min_col_var: float = 1e-8) -> dict:
    """Return per-column variance and participation-ratio effective rank.

    The caller should pass only the rows valid for the shared comparison.
    ``eff_rank_frac`` is diagnostic only; the liveness gate does not use it.
    """
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("features must have at least one row and one column")

    col_var = np.var(X, axis=0)
    eff_rank = _participation_eff_rank(X)
    dim = int(X.shape[1])
    return {
        "col_var": col_var,
        "n_dead_cols": int(np.sum(col_var < float(min_col_var))),
        "min_col_var": float(np.min(col_var)),
        "eff_rank": eff_rank,
        "eff_rank_frac": float(eff_rank / dim),
    }


def multiplex_efficacy(
    features: np.ndarray,
    n_snapshots: int,
    n_obs: int,
) -> dict:
    """Return scale-invariant effective-rank gain from multiplexed blocks.

    This diagnostic is a liveness gate only for built bundles with
    ``n_multiplex > 1``. The pre-registered phase-diagram comparison runs QRC
    at V=1, so the primary QRC path bypasses the multiplex-gain gate. Low gain
    at V>1, especially for QRC virtual nodes at small ``dt``, is expected and
    correctly indicates hollow declared multiplexing; do not force it through
    by lowering the threshold.
    """
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if X.shape[0] == 0:
        raise ValueError("features must have at least one row")
    n_snapshots = int(n_snapshots)
    n_obs = int(n_obs)
    if n_snapshots < 1:
        raise ValueError("n_snapshots must be >= 1")
    if n_obs < 1:
        raise ValueError("n_obs must be >= 1")
    expected_dim = n_snapshots * n_obs
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"features width must equal n_snapshots * n_obs = {expected_dim}; "
            f"got {X.shape[1]}"
        )

    r_full = _participation_eff_rank(X)
    single_ranks = [
        _participation_eff_rank(X[:, k * n_obs : (k + 1) * n_obs])
        for k in range(n_snapshots)
    ]
    r_single_mean = float(np.mean(single_ranks))
    return {
        "r_full": float(r_full),
        "r_single_mean": r_single_mean,
        "gain": float(r_full / max(r_single_mean, 1e-12)),
    }


def assert_features_live(
    bundle: FeatureBundle,
    *,
    min_col_var: float = 1e-8,
    min_multiplex_gain: float = 1.15,
) -> None:
    """Fail on dead columns or ineffective multiplexing.

    The dead-column variance floor always applies. The multiplex-gain gate
    applies only when ``bundle.n_multiplex > 1``. The study's phase-diagram
    comparison runs QRC at V=1; low multiplex gain for exploratory V>1 QRC
    virtual nodes is expected evidence of hollow declared multiplexing, not a
    reason to lower ``min_multiplex_gain``.

    ``eff_rank_frac`` remains a reporting diagnostic from
    :func:`effective_dimension`; it is not a gate criterion.
    """
    if bundle.first_valid < 0:
        raise MatchingError(f"{bundle.model}: first_valid must be non-negative")
    valid = np.asarray(bundle.features, dtype=np.float64)[bundle.first_valid :]
    try:
        diag = effective_dimension(valid, min_col_var=min_col_var)
    except ValueError as exc:
        raise MatchingError(f"{bundle.model}: invalid feature matrix: {exc}") from exc

    dead_cols = int(diag["n_dead_cols"])
    problems: list[str] = []
    if dead_cols:
        problems.append(
            f"dead_features={dead_cols}, min_col_var={diag['min_col_var']:.3g} "
            f"< {float(min_col_var):.3g}"
        )
    if int(bundle.n_multiplex) > 1:
        try:
            efficacy = multiplex_efficacy(valid, bundle.n_multiplex, bundle.n_obs)
        except ValueError as exc:
            raise MatchingError(
                f"{bundle.model}: invalid multiplex layout: {exc}"
            ) from exc
        if efficacy["gain"] < float(min_multiplex_gain):
            problems.append(
                f"multiplex_gain={efficacy['gain']:.3g} "
                f"< {float(min_multiplex_gain):.3g}"
            )
    if problems:
        raise MatchingError(
            f"{bundle.model}: feature-liveness failure: " + "; ".join(problems)
        )


def _hamiltonian_timescale(H_static: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(np.asarray(H_static, dtype=complex))
    bandwidth = float(np.max(eigvals) - np.min(eigvals))
    if bandwidth <= 1e-12:
        raise ValueError("Hamiltonian spectral bandwidth is zero")
    return float(2.0 * np.pi / bandwidth)


def recommend_T_evolve(
    qelm_cfg: QELMConfig,
    coupling_seed: int,
) -> dict:
    """Recommend a QELM snapshot horizon from the shared timescale.

    The returned window is advisory. Feature liveness is checked separately by
    :func:`assert_features_live`, which no longer gates on ``eff_rank_frac``.
    """
    rng = np.random.default_rng(int(coupling_seed))
    J = reservoirs.random_couplings(qelm_cfg.n_qubits, qelm_cfg.J_s, rng)
    H_static = reservoirs.ising_xx_hamiltonian(J, qelm_cfg.h, qelm_cfg.n_qubits)
    jumps = _dissipator_jumps(
        qelm_cfg.dissipator,
        qelm_cfg.n_qubits,
        qelm_cfg.gamma,
    )

    if qelm_cfg.gamma > 0:
        L_base = liouvillian.lindbladian(H_static, jumps)
        gap = diagnostics.spectral_gap(L_base)
        timescale = diagnostics.mixing_time(L_base)
        if not np.isfinite(timescale) or timescale <= 0:
            raise ValueError(f"non-finite dissipative mixing time from gap={gap}")
        regime = "dissipative"
    else:
        timescale = _hamiltonian_timescale(H_static)
        regime = "unitary"

    t_lo = 0.5 * timescale
    return {
        "t_lo": float(t_lo),
        "t_recommended": float(timescale),
        "t_hi": float(2.0 * timescale),
        "regime": regime,
        "timescale": float(timescale),
    }
