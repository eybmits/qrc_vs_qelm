"""Unified feature-builder API for matched QRC/QELM comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from qrc_vs_qelm.core import dissipators, readout, reservoirs
from qrc_vs_qelm.models import qelm


_DISSIPATOR_FACTORIES = {
    "local_loss": dissipators.local_loss,
    "collective_loss": dissipators.collective_loss,
    "dephasing": dissipators.dephasing,
}


def _validate_quantize_levels(levels: int | None) -> None:
    if levels is None:
        return
    if int(levels) != levels:
        raise ValueError("quantize_levels must be an integer, 0, or None")
    if int(levels) == 0:
        return
    if int(levels) < 2:
        raise ValueError("quantize_levels must be >= 2, 0, or None")


def _cache_levels(levels: int | None) -> int | None:
    if levels is None or int(levels) == 0:
        return None
    return int(levels)


def _validate_common(
    *,
    n_qubits: int,
    J_s: float,
    gamma: float,
    dissipator: str,
    quantize_levels: int | None,
) -> None:
    if int(n_qubits) < 1:
        raise ValueError("n_qubits must be >= 1")
    if float(J_s) < 0:
        raise ValueError("J_s must be non-negative")
    if float(gamma) < 0:
        raise ValueError("gamma must be non-negative")
    if dissipator not in _DISSIPATOR_FACTORIES:
        raise ValueError(
            f"unknown dissipator: {dissipator!r}; expected one of "
            f"{sorted(_DISSIPATOR_FACTORIES)}"
        )
    _validate_quantize_levels(quantize_levels)


def _dissipator_jumps(
    name: str,
    n_qubits: int,
    gamma: float,
) -> list[tuple[np.ndarray, float]]:
    _validate_common(
        n_qubits=n_qubits,
        J_s=0.0,
        gamma=gamma,
        dissipator=name,
        quantize_levels=None,
    )
    if float(gamma) == 0.0:
        return []
    return _DISSIPATOR_FACTORIES[name](int(n_qubits), float(gamma))


def _default_observables(n_qubits: int) -> list[readout.Observable]:
    return readout.pauli_observables(
        int(n_qubits),
        axes=("x", "y", "z"),
        max_weight=2,
    )


def _observable_list(
    n_qubits: int,
    observables: Sequence[readout.Observable] | None,
) -> list[readout.Observable]:
    return _default_observables(n_qubits) if observables is None else list(observables)


def _input_vector(inputs: np.ndarray) -> np.ndarray:
    values = np.asarray(inputs, dtype=float)
    if values.ndim != 1:
        raise ValueError("inputs must be a one-dimensional array")
    return values


def quantize_inputs(u: np.ndarray, levels: int | None) -> np.ndarray:
    """Return a copy of ``u`` snapped to a uniform grid on ``[0, 1]``.

    ``levels=0`` or ``None`` leaves values unchanged apart from the defensive
    float64 copy.  Positive quantization uses
    ``round(u * (levels - 1)) / (levels - 1)``.
    """
    values = np.asarray(u, dtype=np.float64)
    if levels is None or int(levels) == 0:
        return values.copy()
    _validate_quantize_levels(levels)
    b = int(levels) - 1
    return (np.round(values * b) / b).astype(np.float64, copy=False)


@dataclass(frozen=True)
class QRCConfig:
    """Configuration for the recurrent Lindblad reservoir feature builder."""

    dt: float
    h_in: float
    n_virtual: int
    n_qubits: int
    J_s: float
    h: float
    gamma: float
    dissipator: str
    quantize_levels: int | None = None

    def __post_init__(self) -> None:
        _validate_common(
            n_qubits=self.n_qubits,
            J_s=self.J_s,
            gamma=self.gamma,
            dissipator=self.dissipator,
            quantize_levels=self.quantize_levels,
        )
        if float(self.dt) <= 0:
            raise ValueError("dt must be positive")
        if int(self.n_virtual) < 1:
            raise ValueError("n_virtual must be >= 1")
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "h_in", float(self.h_in))
        object.__setattr__(self, "n_virtual", int(self.n_virtual))
        object.__setattr__(self, "n_qubits", int(self.n_qubits))
        object.__setattr__(self, "J_s", float(self.J_s))
        object.__setattr__(self, "h", float(self.h))
        object.__setattr__(self, "gamma", float(self.gamma))
        object.__setattr__(self, "quantize_levels", _cache_levels(self.quantize_levels))


@dataclass(frozen=True)
class QELMConfig:
    """Configuration for the static QELM feature builder."""

    d: int
    T_evolve: float
    n_snapshots: int
    n_qubits: int
    J_s: float
    h: float
    gamma: float
    dissipator: str
    quantize_levels: int | None = None

    def __post_init__(self) -> None:
        _validate_common(
            n_qubits=self.n_qubits,
            J_s=self.J_s,
            gamma=self.gamma,
            dissipator=self.dissipator,
            quantize_levels=self.quantize_levels,
        )
        if int(self.d) < 1:
            raise ValueError("d must be >= 1")
        if int(self.d) > int(self.n_qubits):
            raise ValueError("d must satisfy 1 <= d <= n_qubits")
        if float(self.T_evolve) < 0:
            raise ValueError("T_evolve must be non-negative")
        if int(self.n_snapshots) < 1:
            raise ValueError("n_snapshots must be >= 1")
        object.__setattr__(self, "d", int(self.d))
        object.__setattr__(self, "T_evolve", float(self.T_evolve))
        object.__setattr__(self, "n_snapshots", int(self.n_snapshots))
        object.__setattr__(self, "n_qubits", int(self.n_qubits))
        object.__setattr__(self, "J_s", float(self.J_s))
        object.__setattr__(self, "h", float(self.h))
        object.__setattr__(self, "gamma", float(self.gamma))
        object.__setattr__(self, "quantize_levels", _cache_levels(self.quantize_levels))


@dataclass
class FeatureBundle:
    features: np.ndarray
    model: str
    config: dict
    couplings: np.ndarray
    n_obs: int
    n_multiplex: int
    first_valid: int


def build_qrc_features(
    inputs: np.ndarray,
    cfg: QRCConfig,
    coupling_seed: int,
    observables: Sequence[readout.Observable] | None = None,
) -> FeatureBundle:
    """Build full-length recurrent QRC features for ``inputs``."""
    values = _input_vector(inputs)
    obs = _observable_list(cfg.n_qubits, observables)
    rng = np.random.default_rng(int(coupling_seed))
    J = reservoirs.random_couplings(cfg.n_qubits, cfg.J_s, rng)
    H_static = reservoirs.ising_xx_hamiltonian(J, cfg.h, cfg.n_qubits)
    H_drive = cfg.h_in * reservoirs.transverse_drive(cfg.n_qubits)
    jumps = _dissipator_jumps(cfg.dissipator, cfg.n_qubits, cfg.gamma)
    reservoir = reservoirs.LindbladReservoir.from_terms(
        cfg.n_qubits,
        H_static,
        H_drive,
        jumps,
        cfg.dt,
        cache_propagators=cfg.quantize_levels is not None,
        quantize=cfg.quantize_levels,
    )
    driven_inputs = quantize_inputs(values, cfg.quantize_levels)
    features = reservoir.run(
        driven_inputs,
        obs,
        washout=0,
        n_virtual=cfg.n_virtual,
    )
    return FeatureBundle(
        features=np.asarray(features, dtype=np.float64),
        model="qrc",
        config=asdict(cfg),
        couplings=J.copy(),
        n_obs=len(obs),
        n_multiplex=cfg.n_virtual,
        first_valid=0,
    )


def _build_qelm_bundle(
    inputs: np.ndarray,
    cfg: QELMConfig,
    coupling_seed: int,
    observables: Sequence[readout.Observable] | None = None,
) -> FeatureBundle:
    values = _input_vector(inputs)
    obs = _observable_list(cfg.n_qubits, observables)
    encoded_inputs = quantize_inputs(values, cfg.quantize_levels)
    features = qelm.build_qelm_features(
        encoded_inputs,
        n_qubits=cfg.n_qubits,
        d=cfg.d,
        T_evolve=cfg.T_evolve,
        n_snapshots=cfg.n_snapshots,
        gamma=cfg.gamma,
        dissipator=cfg.dissipator,
        J_s=cfg.J_s,
        h=cfg.h,
        coupling_seed=int(coupling_seed),
        observables=obs,
    )
    rng = np.random.default_rng(int(coupling_seed))
    J = reservoirs.random_couplings(cfg.n_qubits, cfg.J_s, rng)
    return FeatureBundle(
        features=np.asarray(features, dtype=np.float64),
        model="qelm",
        config=asdict(cfg),
        couplings=J.copy(),
        n_obs=len(obs),
        n_multiplex=cfg.n_snapshots,
        first_valid=qelm.first_valid_index(cfg.d),
    )


def build_features(
    cfg: QRCConfig | QELMConfig,
    inputs: np.ndarray,
    coupling_seed: int,
    observables: Sequence[readout.Observable] | None = None,
) -> FeatureBundle:
    """Dispatch to the matched QRC or QELM feature builder."""
    if isinstance(cfg, QRCConfig):
        return build_qrc_features(inputs, cfg, coupling_seed, observables)
    if isinstance(cfg, QELMConfig):
        return _build_qelm_bundle(inputs, cfg, coupling_seed, observables)
    raise TypeError(f"unsupported config type: {type(cfg).__name__}")
