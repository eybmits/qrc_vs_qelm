"""Author-requested, separate measured-input QRC-output emulation demonstration."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from qrc_vs_qelm.core import dissipators, readout, reservoirs
from qrc_vs_qelm.matching import assert_matched
from qrc_vs_qelm.models.api import QELMConfig, QRCConfig
from qrc_vs_qelm.models.qelm import QELMChannel, delay_embed
from qrc_vs_qelm.pipeline import _fit_centered_ridge

ROOT = Path(__file__).resolve().parents[2]
DATA_HASH = "d515c7275fb6c7c56a0ebf537d8be2f87f06f3ad36f8153d759519425c746b77"
DATA_URL = (
    "https://raw.githubusercontent.com/shubhomoydas/ad_examples/"
    "81b9fbe053b7da1454202f3e604e25009ded60f3/"
    "datasets/SantaFeCompetition/SF_Acont.dat"
)
ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0)


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = tuple(range(917000, 917012))
    rates: tuple[float, ...] = (4.0, 0.2)
    n_qubits: int = 4
    window: int = 4
    snapshots: int = 4
    dt: float = 0.5
    field: float = 0.5
    levels: int = 33
    block_lengths: tuple[int, ...] = (4096, 1024, 2048)
    washout: int = 128
    prefix: int = 128
    continuation: int = 128


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_laser(path: Path) -> np.ndarray:
    if digest(path) != DATA_HASH:
        raise ValueError("Laser source hash mismatch; do not silently substitute data")
    values = np.loadtxt(path, delimiter=",").reshape(-1)
    if values.size != 9093 or not np.isfinite(values).all():
        raise ValueError("Expected 9093 finite chronological laser samples")
    return values


def input_blocks(values: np.ndarray, cfg: Config) -> tuple[list[np.ndarray], dict]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or values.size < sum(cfg.block_lengths):
        raise ValueError("Insufficient or nonfinite input data")
    end = cfg.block_lengths[0]
    lo, hi = float(values[:end].min()), float(values[:end].max())
    if hi <= lo:
        raise ValueError("Training input is constant")
    cuts = np.r_[0, np.cumsum(cfg.block_lengths)]
    blocks, metadata = [], []
    for left, right in zip(cuts[:-1], cuts[1:]):
        block = values[left:right]
        scaled = (block - lo) / (hi - lo)
        clipped = np.clip(scaled, 0.0, 1.0)
        blocks.append(np.rint(clipped * (cfg.levels - 1)) / (cfg.levels - 1))
        metadata.append({"start": int(left), "stop": int(right),
                         "clipped": int(np.count_nonzero((scaled < 0) | (scaled > 1)))})
    return blocks, {"training_min": lo, "training_max": hi, "blocks": metadata,
                    "levels": cfg.levels, "unused_samples": values.size - int(cuts[-1])}


@dataclass(frozen=True)
class FittedReadout:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    alpha: float
    validation_mse: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.scale) @ self.weights + self.intercept


def fit_readout(x_train, y_train, x_val, y_val) -> FittedReadout:
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    xt, xv = (x_train - mean) / scale, (x_val - mean) / scale
    candidates = []
    for alpha in ALPHAS:
        weights, intercept = _fit_centered_ridge(xt, y_train, alpha)
        error = float(np.mean((xv @ weights + intercept - y_val) ** 2))
        candidates.append(FittedReadout(mean, scale, weights, intercept, alpha, error))
    return min(candidates, key=lambda item: item.validation_mse)


def pair_histories(test: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    p, length = cfg.prefix, cfg.continuation
    if len(test) < 2 * p + length:
        raise ValueError("Test block is too short for the predeclared intervention")
    suffix = test[2 * p:2 * p + length]
    return np.r_[test[:p], suffix], np.r_[test[p:2 * p], suffix]


def pair_check(y_a, y_b, q_a, q_b, window: int) -> dict:
    start = window - 1  # fourth common input is index 3, not index 4
    gap = np.abs(y_a[start:] - y_b[start:]) / 2
    shared_difference = float(np.max(np.abs(q_a[start:] - q_b[start:])))
    worst = np.maximum(np.abs(y_a[start:] - q_a[start:]),
                       np.abs(y_b[start:] - q_b[start:]))
    return {
        "max_shared_prediction_difference": shared_difference,
        "pair_bound_min_slack": float(np.min(worst - gap)),
        "half_gap_at_expiry": float(gap[0]),
        "half_gap_max_first16": float(np.max(gap[:16])),
        "predictions_identical": shared_difference <= 1e-10,
        "pointwise_bound_holds": bool(np.all(worst + 1e-10 >= gap)),
    }


def architecture(seed: int, rate: float, cfg: Config):
    common = dict(n_qubits=cfg.n_qubits, J_s=1.0, h=cfg.field, gamma=rate,
                  dissipator="local_loss", quantize_levels=cfg.levels)
    qrc_config = QRCConfig(**common, dt=cfg.dt, n_virtual=cfg.snapshots, h_in=1.0)
    qelm_config = QELMConfig(**common, d=cfg.window, T_evolve=cfg.dt,
                            n_snapshots=cfg.snapshots, encoding="parallel")
    obs = readout.pauli_observables(cfg.n_qubits, max_weight=1)
    assert_matched(qrc_config, qelm_config, obs)
    coupling = reservoirs.random_couplings(cfg.n_qubits, 1.0, np.random.default_rng(seed))
    hamiltonian = reservoirs.ising_xx_hamiltonian(coupling, cfg.field, cfg.n_qubits)
    jumps = dissipators.local_loss(cfg.n_qubits, rate)
    qrc = reservoirs.LindbladReservoir.from_terms(
        cfg.n_qubits, hamiltonian, reservoirs.transverse_drive(cfg.n_qubits),
        jumps, cfg.dt, quantize=cfg.levels,
    )
    qelm = QELMChannel.from_terms(cfg.n_qubits, hamiltonian, jumps, cfg.dt, cfg.snapshots)
    return qrc, qelm, obs, coupling


def physical_check(qrc, inputs: np.ndarray) -> dict:
    state = qrc.initial_state()
    trace_error, hermitian_error, minimum_eigenvalue = 0.0, 0.0, 1.0
    for value in inputs:
        state = qrc.step(state, float(value))
        trace_error = max(trace_error, float(abs(np.trace(state) - 1)))
        hermitian_error = max(hermitian_error, float(np.max(abs(state - state.conj().T))))
        minimum_eigenvalue = min(minimum_eigenvalue, float(np.linalg.eigvalsh(state).min()))
    return {"max_trace_error": trace_error, "max_hermitian_error": hermitian_error,
            "minimum_eigenvalue": minimum_eigenvalue,
            "passed": trace_error < 1e-10 and hermitian_error < 1e-10
            and minimum_eigenvalue > -1e-10}


def run_cell(blocks, seed: int, rate: float, cfg: Config, output: Path) -> dict:
    start = time.perf_counter()
    qrc, qelm, obs, coupling = architecture(seed, rate, cfg)
    width = cfg.snapshots * len(obs)
    target_column = (cfg.snapshots - 1) * len(obs) + [o.name for o in obs].index("z0")
    targets, features = [], []
    for block in blocks:
        qrc_features = qrc.run(block, obs, n_virtual=cfg.snapshots)
        qelm_features = qelm.features(delay_embed(block, cfg.window), obs,
                                     encoding_scheme="one_to_one")
        if qrc_features.shape != qelm_features.shape or qelm_features.shape[1] != width:
            raise AssertionError("Built resource widths are not matched")
        targets.append(qrc_features[:, target_column])
        features.append(qelm_features)
    valid = slice(max(cfg.washout, cfg.window - 1), None)
    fitted = fit_readout(features[0][valid], targets[0][valid],
                         features[1][valid], targets[1][valid])
    predicted = fitted.predict(features[2])
    train_sd = float(np.std(targets[0][valid]))
    if train_sd <= 1e-12:
        raise ValueError("Constant target: normalized RMSE is not informative")
    rmse = float(np.sqrt(np.mean((predicted[valid] - targets[2][valid]) ** 2)))
    first, second = pair_histories(blocks[2], cfg)
    pair_y, pair_q = [], []
    for history in (first, second):
        pair_y.append(qrc.run(history, obs, n_virtual=cfg.snapshots)[cfg.prefix:, target_column])
        fx = qelm.features(delay_embed(history, cfg.window), obs,
                           encoding_scheme="one_to_one")
        pair_q.append(fitted.predict(fx)[cfg.prefix:])
    checks = pair_check(*pair_y, *pair_q, cfg.window)
    physical = physical_check(qrc, first)
    name = f"seed{seed}_gamma{rate:g}"
    np.savez_compressed(
        output / f"{name}.npz", natural_input=blocks[2], natural_target=targets[2],
        natural_prediction=predicted, pair_input=first[cfg.prefix:],
        pair_y_a=pair_y[0], pair_y_b=pair_y[1], pair_q_a=pair_q[0], pair_q_b=pair_q[1],
        coupling=coupling, readout_mean=fitted.mean, readout_scale=fitted.scale,
        readout_weights=fitted.weights, readout_intercept=fitted.intercept,
    )
    result = {"seed": seed, "gamma": rate, "n_features_each": width,
              "test_scored_rows": len(predicted[valid]), "target": "z0_final_snapshot",
              "test_rmse": rmse, "training_target_sd": train_sd,
              "test_nrmse_training_sd": rmse / train_sd,
              "alpha": fitted.alpha, "validation_mse": fitted.validation_mse,
              "pair": checks, "physical_check": physical,
              "seconds": time.perf_counter() - start, "array_file": f"{name}.npz"}
    dump(output / f"{name}.json", result)
    return result


def run(data: Path, output: Path, cfg: Config) -> dict:
    # Existing results are never overwritten by a new run.
    output.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / "pra_extension/LASER_DEMONSTRATION_PROTOCOL.md"
    before = {"config": asdict(cfg), "protocol_sha256": digest(protocol),
              "source_url": DATA_URL, "data_sha256": digest(data),
              "runner_sha256": digest(Path(__file__)),
              "status": "specified_before_simulation", "scope": "separate_demonstration"}
    dump(output / "run_specification.json", before)
    raw = load_laser(data)
    blocks, preprocessing = input_blocks(raw, cfg)
    dump(output / "preprocessing.json", preprocessing)
    results = []
    for seed in cfg.seeds:
        for rate in cfg.rates:
            result = run_cell(blocks, seed, rate, cfg, output)
            results.append(result)
            print(json.dumps({"seed": seed, "gamma": rate,
                              "nrmse": result["test_nrmse_training_sd"],
                              "pair_half_gap": result["pair"]["half_gap_max_first16"]}), flush=True)
    groups = {}
    for rate in cfg.rates:
        cells = [r for r in results if r["gamma"] == rate]
        groups[str(rate)] = {
            "n_seeds": len(cells),
            "nrmse_median": float(np.median([r["test_nrmse_training_sd"] for r in cells])),
            "nrmse_min_max": [float(f([r["test_nrmse_training_sd"] for r in cells]))
                               for f in (np.min, np.max)],
            "half_gap_max16_median": float(np.median([r["pair"]["half_gap_max_first16"] for r in cells])),
        }
    gates = {
        "identical_windows": all(r["pair"]["predictions_identical"] for r in results),
        "pointwise_pair_bound": all(r["pair"]["pointwise_bound_holds"] for r in results),
        "sampled_physical_states": all(r["physical_check"]["passed"] for r in results),
        "fast_fit_median_below_10_percent": groups["4.0"]["nrmse_median"] < 0.1,
        "slow_pair_gap_median_larger": groups["0.2"]["half_gap_max16_median"] > groups["4.0"]["half_gap_max16_median"],
        "slow_pair_gap_above_001_exists": any(r["pair"]["half_gap_max_first16"] > 0.01 for r in results if r["gamma"] == 0.2),
    }
    summary = {"config": asdict(cfg), "groups": groups, "checks": gates,
               "results": results, "scope": "measured_input_simulation_not_hardware_or_forecasting",
               "files_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
    dump(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/santa_fe_laser/SF_Acont.dat")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.data, args.output, Config())
    print(json.dumps({"groups": summary["groups"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
