#!/usr/bin/env python
"""Exploratory simulator recurrence-fingerprint proof of concept.

This pilot asks an operational question: after two input histories become
identical, does the measured output retain accessible information about the
earlier, different prefixes?  A declared finite-window QELM must have an exact
cutoff once the common suffix covers its window.  A positive tail beyond that
cutoff is a one-sided witness of accessible recurrence; absence of a tail is
reported as window-compatible or inconclusive, never as proof of model identity.

The reset intervention is idealized by construction; reset fidelity is not
tested.  The runner is deliberately separate from the preregistered stage
runners, uses design/pilot coupling seeds only, and produces exploratory output.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrc_vs_qelm import manifests
from qrc_vs_qelm.core import dissipators, readout, reservoirs
from qrc_vs_qelm.matching import assert_matched
from qrc_vs_qelm.models import qelm
from qrc_vs_qelm.models.api import (
    QELMConfig,
    QRCConfig,
    quantize_inputs,
    windowed_qelm_from_qrc,
)
from qrc_vs_qelm.seeds import SEED_GRIDS


INFINITY = "infinity"
QRC_MODEL = "qrc"
_DISSIPATORS = {
    "local_loss": dissipators.local_loss,
    "collective_loss": dissipators.collective_loss,
    "dephasing": dissipators.dephasing,
}


@dataclass(frozen=True)
class QELMSpec:
    encoding: str
    window: int

    def __post_init__(self) -> None:
        if self.encoding not in {"spatial", "time_delayed"}:
            raise ValueError("encoding must be 'spatial' or 'time_delayed'")
        if int(self.window) < 1:
            raise ValueError("QELM window must be >= 1")
        object.__setattr__(self, "window", int(self.window))

    @property
    def model_name(self) -> str:
        prefix = "spatial_qelm" if self.encoding == "spatial" else "replay_qelm"
        return f"{prefix}_d{self.window}"


@dataclass(frozen=True)
class FingerprintProfile:
    name: str
    n_qubits: int
    gamma: float
    dt: float
    h: float
    h_in: float
    J_s: float
    dissipator: str
    quantize_levels: int | None
    burnin_length: int
    branch_length: int
    suffix_lengths: tuple[int, ...]
    pairs_per_coupling: int
    coupling_seeds: tuple[int, ...]
    history_seed: int
    shot_seed: int
    shot_tiers: tuple[int | None, ...]
    n_boot: int
    qelm_specs: tuple[QELMSpec, ...]
    reference_window: int
    tail_margin: float = 0.05
    cutoff_margin: float = 0.02
    reset_suppression_margin: float = 0.80
    exact_floor: float = 1e-14

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if self.burnin_length < 0:
            raise ValueError("burnin_length must be >= 0")
        if self.branch_length < 1:
            raise ValueError("branch_length must be >= 1")
        if self.pairs_per_coupling < 1:
            raise ValueError("pairs_per_coupling must be >= 1")
        if self.n_boot < 1:
            raise ValueError("n_boot must be >= 1")
        if not self.coupling_seeds:
            raise ValueError("at least one coupling seed is required")
        if not self.qelm_specs:
            raise ValueError("at least one QELM specification is required")
        if self.dissipator not in _DISSIPATORS:
            raise ValueError(f"unknown dissipator: {self.dissipator!r}")
        suffixes = tuple(int(value) for value in self.suffix_lengths)
        if suffixes != tuple(sorted(set(suffixes))):
            raise ValueError("suffix_lengths must be unique and sorted")
        if not suffixes or suffixes[0] != 0:
            raise ValueError("suffix_lengths must start at 0")
        required = {int(self.reference_window)} | {
            int(spec.window) for spec in self.qelm_specs
        }
        if not required.issubset(suffixes):
            raise ValueError("suffix_lengths must contain every declared window")
        if self.branch_length < max(spec.window for spec in self.qelm_specs):
            raise ValueError("branch_length must cover the largest QELM window")
        for spec in self.qelm_specs:
            if spec.encoding == "spatial" and spec.window > self.n_qubits:
                raise ValueError("spatial QELM windows cannot exceed n_qubits")
        for tier in self.shot_tiers:
            if tier is not None and int(tier) < 1:
                raise ValueError("finite shot tiers must be >= 1")


_PILOT_COUPLINGS = tuple(
    dict.fromkeys(cell.coupling_seed for cell in SEED_GRIDS["pilot"])
)

PROFILES: dict[str, FingerprintProfile] = {
    "tiny": FingerprintProfile(
        name="tiny",
        n_qubits=2,
        gamma=0.3,
        dt=0.5,
        h=0.5,
        h_in=1.0,
        J_s=1.0,
        dissipator="local_loss",
        quantize_levels=17,
        burnin_length=4,
        branch_length=6,
        suffix_lengths=(0, 1, 2, 3, 4, 6),
        pairs_per_coupling=4,
        coupling_seeds=(_PILOT_COUPLINGS[0],),
        history_seed=2026082701,
        shot_seed=2026082702,
        shot_tiers=(None, 10_000),
        n_boot=64,
        qelm_specs=(
            QELMSpec("spatial", 2),
            QELMSpec("time_delayed", 3),
        ),
        reference_window=3,
    ),
    "smoke": FingerprintProfile(
        name="smoke",
        n_qubits=3,
        gamma=0.3,
        dt=0.5,
        h=0.5,
        h_in=1.0,
        J_s=1.0,
        dissipator="local_loss",
        quantize_levels=33,
        burnin_length=12,
        branch_length=16,
        suffix_lengths=(0, 1, 2, 3, 4, 5, 6, 8, 12, 16),
        pairs_per_coupling=24,
        coupling_seeds=_PILOT_COUPLINGS,
        history_seed=2026082711,
        shot_seed=2026082712,
        shot_tiers=(None, 10_000, 1_000),
        n_boot=500,
        qelm_specs=(
            QELMSpec("spatial", 3),
            QELMSpec("time_delayed", 5),
        ),
        reference_window=5,
    ),
    "full": FingerprintProfile(
        name="full",
        n_qubits=5,
        gamma=0.3,
        dt=0.5,
        h=0.5,
        h_in=1.0,
        J_s=1.0,
        dissipator="local_loss",
        quantize_levels=33,
        burnin_length=100,
        branch_length=40,
        suffix_lengths=(0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32),
        pairs_per_coupling=100,
        coupling_seeds=_PILOT_COUPLINGS,
        history_seed=2026082721,
        shot_seed=2026082722,
        shot_tiers=(None, 100_000, 10_000, 1_000, 100),
        n_boot=2_000,
        qelm_specs=(
            QELMSpec("spatial", 5),
            QELMSpec("time_delayed", 5),
        ),
        reference_window=5,
    ),
}


def _profile_payload(profile: FingerprintProfile) -> dict[str, Any]:
    payload = asdict(profile)
    payload["shot_tiers"] = [
        INFINITY if tier is None else int(tier) for tier in profile.shot_tiers
    ]
    return payload


def _shot_label(tier: int | None) -> str:
    return INFINITY if tier is None else str(int(tier))


def _seeded_rng(seed: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, parts)]))


def _observables(profile: FingerprintProfile) -> list[readout.Observable]:
    return readout.pauli_observables(
        profile.n_qubits,
        axes=("x", "y", "z"),
        max_weight=2,
    )


def _configs(
    profile: FingerprintProfile,
    observables: list[readout.Observable],
) -> tuple[QRCConfig, dict[str, QELMConfig]]:
    qrc_cfg = QRCConfig(
        dt=profile.dt,
        h_in=profile.h_in,
        n_virtual=1,
        n_qubits=profile.n_qubits,
        J_s=profile.J_s,
        h=profile.h,
        gamma=profile.gamma,
        dissipator=profile.dissipator,
        quantize_levels=profile.quantize_levels,
    )
    qelm_cfgs: dict[str, QELMConfig] = {}
    for spec in profile.qelm_specs:
        if spec.encoding == "time_delayed":
            cfg = windowed_qelm_from_qrc(qrc_cfg, window_k=spec.window)
        else:
            cfg = QELMConfig(
                d=spec.window,
                T_evolve=profile.dt,
                n_snapshots=1,
                n_qubits=profile.n_qubits,
                J_s=profile.J_s,
                h=profile.h,
                gamma=profile.gamma,
                dissipator=profile.dissipator,
                quantize_levels=profile.quantize_levels,
                h_in=profile.h_in,
                encoding="spatial",
            )
        assert_matched(qrc_cfg, cfg, observables=observables)
        qelm_cfgs[spec.model_name] = cfg
    return qrc_cfg, qelm_cfgs


def _build_dynamics(
    profile: FingerprintProfile,
    coupling_seed: int,
) -> tuple[
    reservoirs.LindbladReservoir,
    qelm.QELMChannel,
    list[readout.Observable],
]:
    obs = _observables(profile)
    rng = np.random.default_rng(int(coupling_seed))
    couplings = reservoirs.random_couplings(profile.n_qubits, profile.J_s, rng)
    h_static = reservoirs.ising_xx_hamiltonian(
        couplings,
        profile.h,
        profile.n_qubits,
    )
    h_drive = profile.h_in * reservoirs.transverse_drive(profile.n_qubits)
    jumps = (
        []
        if profile.gamma == 0
        else _DISSIPATORS[profile.dissipator](profile.n_qubits, profile.gamma)
    )
    reservoir = reservoirs.LindbladReservoir.from_terms(
        profile.n_qubits,
        h_static,
        h_drive,
        jumps,
        profile.dt,
        cache_propagators=profile.quantize_levels is not None,
        quantize=profile.quantize_levels,
    )
    spatial_channel = qelm.QELMChannel.from_terms(
        n_qubits=profile.n_qubits,
        H_static=h_static,
        jumps=jumps,
        T_evolve=profile.dt,
        n_snapshots=1,
    )
    return reservoir, spatial_channel, obs


def _state_feature(
    rho: np.ndarray,
    observables: list[readout.Observable],
) -> np.ndarray:
    return readout.features_from_states([rho], observables).reshape(-1)


def _evolve(
    reservoir: reservoirs.LindbladReservoir,
    rho: np.ndarray,
    inputs: Iterable[float],
) -> np.ndarray:
    state = rho.copy()
    for value in inputs:
        state = reservoir.step(state, float(value))
    return state


def _recent_window(history: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if len(values) < int(window):
        raise ValueError("history is shorter than the declared QELM window")
    return values[-int(window):]


def _qelm_feature_pair(
    *,
    encoding: str,
    window: int,
    history_a: np.ndarray,
    history_b: np.ndarray,
    reservoir: reservoirs.LindbladReservoir,
    spatial_channel: qelm.QELMChannel,
    observables: list[readout.Observable],
) -> tuple[np.ndarray, np.ndarray]:
    recent_a = _recent_window(history_a, window)
    recent_b = _recent_window(history_b, window)
    if encoding == "time_delayed":
        states_a = qelm.time_delayed_window_states(reservoir, recent_a, 1)
        states_b = qelm.time_delayed_window_states(reservoir, recent_b, 1)
        return (
            readout.features_from_states(states_a, observables).reshape(-1),
            readout.features_from_states(states_b, observables).reshape(-1),
        )

    # Existing spatial-QELM semantics encode newest input first.
    windows = np.stack([recent_a[::-1], recent_b[::-1]])
    features = spatial_channel.features(windows, observables)
    return features[0].copy(), features[1].copy()


def _exact_curves_for_pair(
    profile: FingerprintProfile,
    *,
    coupling_seed: int,
    pair_id: int,
    reservoir: reservoirs.LindbladReservoir,
    spatial_channel: qelm.QELMChannel,
    observables: list[readout.Observable],
) -> dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]]:
    rng = _seeded_rng(profile.history_seed, coupling_seed, pair_id)
    common = quantize_inputs(
        rng.uniform(0.0, 1.0, size=profile.burnin_length),
        profile.quantize_levels,
    )
    branch_a = quantize_inputs(
        rng.uniform(0.0, 1.0, size=profile.branch_length),
        profile.quantize_levels,
    )
    branch_b = quantize_inputs(
        rng.uniform(0.0, 1.0, size=profile.branch_length),
        profile.quantize_levels,
    )
    suffix = quantize_inputs(
        rng.uniform(0.0, 1.0, size=max(profile.suffix_lengths)),
        profile.quantize_levels,
    )

    initial = reservoir.initial_state()
    common_state = _evolve(reservoir, initial, common)
    carry_a = _evolve(reservoir, common_state, branch_a)
    carry_b = _evolve(reservoir, common_state, branch_b)
    reset_state = reservoir.initial_state()

    history_a = np.concatenate([common, branch_a])
    history_b = np.concatenate([common, branch_b])
    records: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    wanted = set(profile.suffix_lengths)

    for length in range(max(profile.suffix_lengths) + 1):
        if length in wanted:
            records[(QRC_MODEL, "carry", length)] = (
                _state_feature(carry_a, observables),
                _state_feature(carry_b, observables),
            )
            reset_feature = _state_feature(reset_state, observables)
            records[(QRC_MODEL, "ideal_quantum_reset", length)] = (
                reset_feature.copy(),
                reset_feature.copy(),
            )
            extended_a = np.concatenate([history_a, suffix[:length]])
            extended_b = np.concatenate([history_b, suffix[:length]])
            for spec in profile.qelm_specs:
                pair = _qelm_feature_pair(
                    encoding=spec.encoding,
                    window=spec.window,
                    history_a=extended_a,
                    history_b=extended_b,
                    reservoir=reservoir,
                    spatial_channel=spatial_channel,
                    observables=observables,
                )
                # A quantum-only reset is a no-op for a reset-per-row QELM when
                # its explicit classical delay window is preserved.
                records[(spec.model_name, "carry", length)] = pair
                records[(spec.model_name, "ideal_quantum_reset", length)] = (
                    pair[0].copy(),
                    pair[1].copy(),
                )
        if length == max(profile.suffix_lengths):
            break
        value = float(suffix[length])
        carry_a = reservoir.step(carry_a, value)
        carry_b = reservoir.step(carry_b, value)
        reset_state = reservoir.step(reset_state, value)
    return records


def _cross_batch_estimate(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    tier: int | None,
    seed: int,
    parts: tuple[int, ...],
) -> tuple[float, float]:
    exact = float(np.mean((x_a - x_b) ** 2))
    if tier is None:
        return exact, exact

    noisy: list[np.ndarray] = []
    for batch in range(2):
        for arm, vector in enumerate((x_a, x_b)):
            rng = _seeded_rng(seed, *parts, int(tier), batch, arm)
            noisy.append(_sample_pauli_expectations(vector, int(tier), rng))
    diff_1 = noisy[0] - noisy[1]
    diff_2 = noisy[2] - noisy[3]
    estimate = float(np.mean(diff_1 * diff_2))
    raw_mse = float(0.5 * (np.mean(diff_1**2) + np.mean(diff_2**2)))
    return estimate, raw_mse


def _sample_pauli_expectations(
    expectations: np.ndarray,
    n_shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample exact +-1 Pauli means using independent binomial counts."""
    if int(n_shots) < 1:
        raise ValueError("n_shots must be >= 1")
    values = np.clip(np.asarray(expectations, dtype=float), -1.0, 1.0)
    plus_probability = 0.5 * (1.0 + values)
    plus_counts = rng.binomial(int(n_shots), plus_probability)
    return (2.0 * plus_counts / int(n_shots) - 1.0).astype(float)


def run_pairs(profile: FingerprintProfile) -> pd.DataFrame:
    """Run all paired histories and return stable long-form estimands."""
    observables = _observables(profile)
    _configs(profile, observables)
    model_windows = {QRC_MODEL: int(profile.reference_window)}
    model_encodings = {QRC_MODEL: "recurrent"}
    for spec in profile.qelm_specs:
        model_windows[spec.model_name] = spec.window
        model_encodings[spec.model_name] = spec.encoding

    rows: list[dict[str, Any]] = []
    model_codes = {name: idx for idx, name in enumerate(model_windows)}
    condition_codes = {"carry": 0, "ideal_quantum_reset": 1}
    suffix_codes = {value: idx for idx, value in enumerate(profile.suffix_lengths)}

    for coupling_seed in profile.coupling_seeds:
        reservoir, channel, live_observables = _build_dynamics(profile, coupling_seed)
        if [item.name for item in observables] != [item.name for item in live_observables]:
            raise RuntimeError("observable order changed during dynamics construction")
        for pair_id in range(profile.pairs_per_coupling):
            curves = _exact_curves_for_pair(
                profile,
                coupling_seed=coupling_seed,
                pair_id=pair_id,
                reservoir=reservoir,
                spatial_channel=channel,
                observables=observables,
            )
            for (model, condition, suffix_length), (x_a, x_b) in curves.items():
                if x_a.shape != x_b.shape or x_a.shape != (len(observables),):
                    raise RuntimeError("fingerprint feature dimension mismatch")
                exact = float(np.mean((x_a - x_b) ** 2))
                for tier in profile.shot_tiers:
                    estimate, raw_mse = _cross_batch_estimate(
                        x_a,
                        x_b,
                        tier=tier,
                        seed=profile.shot_seed,
                        parts=(
                            int(coupling_seed),
                            int(pair_id),
                            int(model_codes[model]),
                            int(condition_codes[condition]),
                            int(suffix_codes[suffix_length]),
                        ),
                    )
                    rows.append(
                        {
                            "coupling_seed": int(coupling_seed),
                            "pair_id": int(pair_id),
                            "model": model,
                            "encoding": model_encodings[model],
                            "declared_window": int(model_windows[model]),
                            "condition": condition,
                            "suffix_length": int(suffix_length),
                            "shots": _shot_label(tier),
                            "n_observables": int(len(observables)),
                            "exact_signal_mse": exact,
                            "cross_batch_signal_mse": estimate,
                            "raw_two_batch_mse": raw_mse,
                        }
                    )
    frame = pd.DataFrame(rows)
    sort_cols = [
        "model",
        "condition",
        "shots",
        "suffix_length",
        "coupling_seed",
        "pair_id",
    ]
    return frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def summarize_curves(
    scores: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize curves, bootstrapping independent coupling configurations."""
    keys = [
        "model",
        "encoding",
        "declared_window",
        "condition",
        "suffix_length",
        "shots",
    ]
    coupling = (
        scores.groupby(keys + ["coupling_seed"], sort=True, as_index=False)
        .agg(
            signal_mse=("cross_batch_signal_mse", "mean"),
            exact_signal_mse=("exact_signal_mse", "mean"),
            raw_two_batch_mse=("raw_two_batch_mse", "mean"),
            n_pairs=("pair_id", "nunique"),
        )
    )
    rows: list[dict[str, Any]] = []
    for group_index, (group_key, group) in enumerate(coupling.groupby(keys, sort=True)):
        values = group["signal_mse"].to_numpy(dtype=float)
        rng = _seeded_rng(seed, group_index)
        if len(values) == 1:
            boot = np.repeat(values[0], int(n_boot))
        else:
            picks = rng.integers(0, len(values), size=(int(n_boot), len(values)))
            boot = values[picks].mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        row = dict(zip(keys, group_key, strict=True))
        row.update(
            {
                "n_couplings": int(len(values)),
                "pairs_per_coupling": int(group["n_pairs"].min()),
                "signal_mse_mean": float(values.mean()),
                "signal_mse_ci_low": float(low),
                "signal_mse_ci_high": float(high),
                "exact_signal_mse_mean": float(group["exact_signal_mse"].mean()),
                "raw_two_batch_mse_mean": float(group["raw_two_batch_mse"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys, kind="mergesort").reset_index(drop=True)


def _classification_indices(
    scores: pd.DataFrame,
    profile: FingerprintProfile,
) -> pd.DataFrame:
    """Compute normalized indices with a paired hierarchical bootstrap.

    Coupling configurations are the outer generalization units.  Within each
    resampled coupling, complete history-pair trajectories are resampled so all
    suffix lengths and both reset conditions remain paired.
    """

    suffixes = np.asarray(profile.suffix_lengths, dtype=int)
    condition_order = ("carry", "ideal_quantum_reset")

    def metric_vector(curve: np.ndarray, window: int) -> np.ndarray:
        carry = curve[0]
        reset = curve[1]
        baseline = float(carry[0])
        if baseline <= profile.exact_floor:
            return np.full(6, np.nan, dtype=float)
        pre_mask = suffixes < int(window)
        tail_mask = suffixes >= int(window)
        ratios = carry / baseline
        p_index = float(np.max(ratios[pre_mask]))
        t_index = float(np.mean(ratios[tail_mask]))
        max_tail = float(np.max(ratios[tail_mask]))
        t_reset = float(np.mean(reset[tail_mask]) / baseline)
        suppression = (
            float(1.0 - t_reset / t_index)
            if t_index > profile.tail_margin
            else float("nan")
        )
        return np.asarray(
            [baseline, p_index, t_index, max_tail, t_reset, suppression],
            dtype=float,
        )

    metric_names = (
        "baseline_signal_mse",
        "precutoff_liveness_P",
        "beyond_window_tail_T",
        "max_beyond_window_R",
        "reset_tail_T",
        "reset_suppression_S",
    )
    rows: list[dict[str, Any]] = []
    for group_index, ((model, shots_label), group) in enumerate(
        scores.groupby(["model", "shots"], sort=True)
    ):
        window = int(group["declared_window"].iloc[0])
        coupling_seeds = sorted(group["coupling_seed"].unique())
        pair_ids = sorted(group["pair_id"].unique())
        values = np.empty(
            (
                len(coupling_seeds),
                len(pair_ids),
                len(condition_order),
                len(suffixes),
            ),
            dtype=float,
        )
        for coupling_index, coupling_seed in enumerate(coupling_seeds):
            for pair_index, pair_id in enumerate(pair_ids):
                unit = group[
                    (group["coupling_seed"] == coupling_seed)
                    & (group["pair_id"] == pair_id)
                ]
                if len(unit) != len(condition_order) * len(suffixes):
                    raise RuntimeError("incomplete paired fingerprint trajectory")
                for condition_index, condition in enumerate(condition_order):
                    selected = unit[unit["condition"] == condition].set_index(
                        "suffix_length"
                    )
                    values[coupling_index, pair_index, condition_index] = selected.loc[
                        suffixes,
                        "cross_batch_signal_mse",
                    ].to_numpy(float)

        point = metric_vector(values.mean(axis=(0, 1)), window)
        rng = _seeded_rng(profile.history_seed + 173, group_index)
        boot = np.empty((profile.n_boot, len(metric_names)), dtype=float)
        for boot_index in range(profile.n_boot):
            selected_couplings = rng.integers(
                0,
                len(coupling_seeds),
                size=len(coupling_seeds),
            )
            configuration_curves = []
            for coupling_index in selected_couplings:
                selected_pairs = rng.integers(0, len(pair_ids), size=len(pair_ids))
                configuration_curves.append(
                    values[coupling_index, selected_pairs].mean(axis=0)
                )
            boot_curve = np.mean(configuration_curves, axis=0)
            boot[boot_index] = metric_vector(boot_curve, window)

        lows = np.empty(len(metric_names), dtype=float)
        highs = np.empty(len(metric_names), dtype=float)
        for metric_index in range(len(metric_names)):
            finite = boot[np.isfinite(boot[:, metric_index]), metric_index]
            if finite.size:
                lows[metric_index], highs[metric_index] = np.percentile(
                    finite,
                    [2.5, 97.5],
                )
            else:
                lows[metric_index] = np.nan
                highs[metric_index] = np.nan
        metrics = dict(zip(metric_names, point, strict=True))
        ci_lows = dict(zip(metric_names, lows, strict=True))
        ci_highs = dict(zip(metric_names, highs, strict=True))
        live = bool(ci_lows["baseline_signal_mse"] > profile.exact_floor)
        positive_tail = bool(
            live
            and ci_lows["beyond_window_tail_T"] > profile.tail_margin
        )
        reset_removed = bool(
            positive_tail
            and ci_lows["reset_suppression_S"] > profile.reset_suppression_margin
            and ci_highs["reset_tail_T"] < profile.cutoff_margin
        )
        cutoff_compatible = bool(
            live
            and ci_lows["precutoff_liveness_P"] > 0.10
            and ci_highs["max_beyond_window_R"] < profile.cutoff_margin
        )
        if positive_tail and reset_removed:
            classification = "RECURRENT_MEMORY_DETECTED"
        elif cutoff_compatible:
            classification = "FINITE_WINDOW_COMPATIBLE"
        else:
            classification = "INDETERMINATE"
        rows.append(
            {
                "model": model,
                "shots": shots_label,
                "declared_window": window,
                "n_couplings": int(len(coupling_seeds)),
                "pairs_per_coupling": int(len(pair_ids)),
                **metrics,
                **{
                    f"{name}_ci_low": ci_lows[name]
                    for name in metric_names
                },
                **{
                    f"{name}_ci_high": ci_highs[name]
                    for name in metric_names
                },
                "classification": classification,
                "positive_tail": positive_tail,
                "reset_removed": reset_removed,
                "cutoff_compatible": cutoff_compatible,
            }
        )
    return pd.DataFrame(rows).sort_values(["shots", "model"], kind="mergesort")


def _verdict(indices: pd.DataFrame, profile: FingerprintProfile) -> dict[str, Any]:
    exact = indices[indices["shots"] == INFINITY].set_index("model")
    qrc_exact = str(exact.loc[QRC_MODEL, "classification"])
    qelm_names = [spec.model_name for spec in profile.qelm_specs]
    qelm_exact = {name: str(exact.loc[name, "classification"]) for name in qelm_names}
    exact_pass = bool(
        qrc_exact == "RECURRENT_MEMORY_DETECTED"
        and all(value == "FINITE_WINDOW_COMPATIBLE" for value in qelm_exact.values())
    )
    finite_labels = sorted(
        (int(label) for label in indices["shots"].unique() if label != INFINITY),
        reverse=True,
    )
    robust_label = str(finite_labels[0]) if finite_labels else None
    robust_pass = False
    robust_classes: dict[str, str] = {}
    if robust_label is not None:
        finite = indices[indices["shots"] == robust_label].set_index("model")
        robust_classes = {
            name: str(finite.loc[name, "classification"])
            for name in [QRC_MODEL, *qelm_names]
        }
        robust_pass = bool(
            robust_classes[QRC_MODEL] == "RECURRENT_MEMORY_DETECTED"
            and all(
                robust_classes[name] == "FINITE_WINDOW_COMPATIBLE"
                for name in qelm_names
            )
        )
    if exact_pass and robust_pass:
        status = "POC_SMOKE_PASS_EXACT_AND_FINITE_SHOTS"
    elif exact_pass:
        status = "POC_SMOKE_PASS_EXACT_ONLY"
    else:
        status = "POC_INCONCLUSIVE"
    return {
        "status": status,
        "exact_pass": exact_pass,
        "finite_shot_pass": robust_pass,
        "finite_shot_tier": robust_label,
        "exact_classifications": {QRC_MODEL: qrc_exact, **qelm_exact},
        "finite_shot_classifications": robust_classes,
        "interpretation": (
            "A positive, reset-sensitive suffix tail witnesses accessible recurrent "
            "input memory beyond the declared QELM window. A hard cutoff is only "
            "finite-window-compatible; it does not prove source-code identity. "
            "The simulator reset is idealized and does not certify reset fidelity."
        ),
    }


def plot_fingerprint(
    summary: pd.DataFrame,
    profile: FingerprintProfile,
    out_path: Path,
) -> None:
    exact = summary[
        (summary["shots"] == INFINITY) & (summary["condition"] == "carry")
    ]
    finite_tiers = sorted(
        (int(value) for value in summary["shots"].unique() if value != INFINITY),
        reverse=True,
    )
    finite_label = str(finite_tiers[0]) if finite_tiers else None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    colors = {
        QRC_MODEL: "#b22222",
        "spatial": "#2166ac",
        "time_delayed": "#4d9221",
    }
    for model, group in exact.groupby("model", sort=True):
        encoding = str(group["encoding"].iloc[0])
        color = colors[QRC_MODEL] if model == QRC_MODEL else colors[encoding]
        x = group["suffix_length"].to_numpy(int)
        y = np.sqrt(np.maximum(group["signal_mse_mean"].to_numpy(float), 1e-18))
        low = np.sqrt(np.maximum(group["signal_mse_ci_low"].to_numpy(float), 1e-18))
        high = np.sqrt(np.maximum(group["signal_mse_ci_high"].to_numpy(float), 1e-18))
        axes[0].plot(x, y, marker="o", linewidth=2, label=model, color=color)
        axes[0].fill_between(x, low, high, color=color, alpha=0.16)
    qrc_reset_exact = summary[
        (summary["shots"] == INFINITY)
        & (summary["condition"] == "ideal_quantum_reset")
        & (summary["model"] == QRC_MODEL)
    ]
    if not qrc_reset_exact.empty:
        axes[0].plot(
            qrc_reset_exact["suffix_length"],
            np.sqrt(
                np.maximum(qrc_reset_exact["signal_mse_mean"].to_numpy(float), 1e-18)
            ),
            linestyle="--",
            linewidth=1.7,
            color="#666666",
            label="qrc after reset",
        )
    for spec in profile.qelm_specs:
        axes[0].axvline(spec.window, color="#999999", linestyle=":", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Length of identical suffix")
    axes[0].set_ylabel("Exact observable RMS separation")
    axes[0].set_title("Hard cutoff versus recurrent tail")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].text(
        0.98,
        0.03,
        "Exact zeros displayed at $10^{-9}$",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    if finite_label is not None:
        finite = summary[
            (summary["shots"] == finite_label)
            & (summary["condition"] == "carry")
        ]
        for model, group in finite.groupby("model", sort=True):
            encoding = str(group["encoding"].iloc[0])
            color = colors[QRC_MODEL] if model == QRC_MODEL else colors[encoding]
            x = group["suffix_length"].to_numpy(int)
            y = group["signal_mse_mean"].to_numpy(float)
            low = group["signal_mse_ci_low"].to_numpy(float)
            high = group["signal_mse_ci_high"].to_numpy(float)
            axes[1].plot(x, y, marker="o", linewidth=2, label=model, color=color)
            axes[1].fill_between(x, low, high, color=color, alpha=0.16)
        qrc_reset_finite = summary[
            (summary["shots"] == finite_label)
            & (summary["condition"] == "ideal_quantum_reset")
            & (summary["model"] == QRC_MODEL)
        ]
        if not qrc_reset_finite.empty:
            x = qrc_reset_finite["suffix_length"].to_numpy(int)
            y = qrc_reset_finite["signal_mse_mean"].to_numpy(float)
            low = qrc_reset_finite["signal_mse_ci_low"].to_numpy(float)
            high = qrc_reset_finite["signal_mse_ci_high"].to_numpy(float)
            axes[1].plot(
                x,
                y,
                linestyle="--",
                linewidth=1.7,
                color="#666666",
                label="qrc after reset",
            )
            axes[1].fill_between(x, low, high, color="#777777", alpha=0.12)
        axes[1].axhline(0.0, color="black", linewidth=1)
        axes[1].set_title(f"Noise-corrected witness at {int(finite_label):,} shots")
        axes[1].set_ylabel("Cross-batch separation estimate")
        axes[1].legend(fontsize=8)
    else:
        axes[1].axis("off")
    axes[1].set_xlabel("Length of identical suffix")
    axes[1].grid(alpha=0.25)
    fig.suptitle(
        "Exploratory recurrence fingerprint POC — not confirmatory evidence",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _write_outputs(
    *,
    out_dir: Path,
    profile: FingerprintProfile,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    indices: pd.DataFrame,
    verdict: dict[str, Any],
    allow_dirty: bool,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "scores": out_dir / "recurrence_fingerprint_scores.csv",
        "summary": out_dir / "recurrence_fingerprint_summary.csv",
        "indices": out_dir / "recurrence_fingerprint_indices.csv",
        "figure": out_dir / "recurrence_fingerprint.png",
        "figure_pdf": out_dir / "recurrence_fingerprint.pdf",
        "result": out_dir / "recurrence_fingerprint.json",
    }
    scores.to_csv(paths["scores"], index=False, float_format="%.17g")
    summary.to_csv(paths["summary"], index=False, float_format="%.17g")
    indices.to_csv(paths["indices"], index=False, float_format="%.17g")
    plot_fingerprint(summary, profile, paths["figure"])
    source_files = {
        "runner": Path(__file__).resolve(),
        "tests": ROOT / "tests" / "test_recurrence_fingerprint_poc.py",
    }
    source_sha256 = {
        name: manifests.sha256_file(path)
        for name, path in source_files.items()
        if path.exists()
    }
    result_payload = {
        "pilot_only": True,
        "confirmatory_use": "forbidden",
        "confirmatory_seeds_touched": False,
        "reset_semantics": "ideal simulator reset; reset fidelity is not tested",
        "shot_model": "independent exact-binomial Pauli expectation sampling",
        "source_sha256": source_sha256,
        "profile": _profile_payload(profile),
        "verdict": verdict,
        "artifacts": {name: str(path) for name, path in paths.items() if name != "result"},
    }
    paths["result"].write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {
        name: manifests.sha256_file(path)
        for name, path in paths.items()
        if path.exists()
    }
    manifest_path = manifests.write_manifest(
        out_dir,
        "recurrence_fingerprint",
        {
            "repo_root": ROOT,
            "pilot_only": True,
            "confirmatory_use": "forbidden",
            "confirmatory_seeds_touched": False,
            "reset_semantics": "ideal simulator reset; reset fidelity is not tested",
            "shot_model": "independent exact-binomial Pauli expectation sampling",
            "source_sha256": source_sha256,
            "allow_dirty": bool(allow_dirty),
            "profile": _profile_payload(profile),
            "verdict": verdict,
            "artifact_sha256": hashes,
        },
    )
    paths["manifest"] = manifest_path
    return paths


def run(profile: FingerprintProfile) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scores = run_pairs(profile)
    summary = summarize_curves(scores, n_boot=profile.n_boot, seed=profile.history_seed + 91)
    indices = _classification_indices(scores, profile)
    verdict = _verdict(indices, profile)
    return scores, summary, indices, verdict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results_recurrence_fingerprint_poc_smoke",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow the exploratory run on a dirty tree and record the violation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = PROFILES[args.profile]
    dirty_state = manifests.require_clean_tree(bool(args.allow_dirty))
    provenance = {
        "dirty_tree_recorded": bool(dirty_state["preregistration_violated"]),
    }
    scores, summary, indices, verdict = run(profile)
    paths = _write_outputs(
        out_dir=args.out,
        profile=profile,
        scores=scores,
        summary=summary,
        indices=indices,
        verdict={**verdict, **provenance},
        allow_dirty=bool(args.allow_dirty),
    )
    print(
        json.dumps(
            {
                "status": verdict["status"],
                "profile": profile.name,
                "scores": len(scores),
                "output": str(args.out),
                "figure": str(paths["figure"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
